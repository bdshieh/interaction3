
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
import pandas as pd
from itertools import repeat
from contextlib import closing
from tqdm import tqdm

import interaction3.abstract as abstract
from interaction3.mfield.simulations import MultiArrayTransmitReceiveBeamplot
from interaction3.mfield.simulations import sim_functions as sim

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

defaults = dict()
defaults['threads'] = multiprocessing.cpu_count()
defaults['specification'] = None
defaults['file'] = None
defaults['rotations'] = None


## PROCESS FUNCTIONS ##

def init_process(_write_lock):

    global write_lock
    write_lock = _write_lock


def process(args):

    file, field_pos, rotation_rule, sim, arrays = args

    sim = abstract.loads(sim)
    arrays = abstract.loads(arrays)

    # set focus delays while arrays are flat
    tx_focus = sim['transmit_focus']
    rx_focus = sim['receive_focus']
    c = sim['sound_speed']
    quant = sim['delay_quantization']

    for array in arrays:
        abstract.focus_array(array, tx_focus, sound_speed=c, quantization=quant, kind='tx')
        abstract.focus_array(array, rx_focus, sound_speed=c, quantization=quant, kind='rx')

    # rotate arrays based on rotation rule
    for array_id, dir, angle in rotation_rule:

        if dir.lower() == 'x':
            vec = [1, 0, 0]
        elif dir.lower() == 'y':
            vec = [0, 1, 0]
        elif dir.lower() == 'z':
            vec = [0, 0, 1]

        for array in arrays:
            if array['id'] == array_id:
                abstract.rotate_array(array, vec, np.deg2rad(angle))

    # create and run simulation
    kwargs, meta = MultiArrayTransmitReceiveBeamplot.connector(sim, *arrays)
    simulation = MultiArrayTransmitReceiveBeamplot(**kwargs)
    simulation.solve(field_pos)

    # extract results and save
    rf_data = simulation.result['rf_data']
    times = simulation.result['times']

    with write_lock:
        with closing(sql.connect(file)) as con:

            if sim['save_image_data_only']:
                pass
            else:
                for rf, t in zip(rf_data, times):
                    update_pressures_table(con, field_pos, t, rf)
            update_progress(con, field_pos)


## ENTRY POINT ##

def main(**args):

    spec = args['specification']

    objects = MultiArrayTransmitReceiveBeamplot.get_objects_from_spec(*spec)
    simulation = objects[0]
    arrays = objects[1:]

    for k, v in simulation.items():
        args.setdefault(k, v)
    for k, v in defaults.items():
        args.setdefault(k, v)

    threads = args['threads']
    file = args['file']
    rotations = args['rotation']

    mode = simulation['mesh_mode']
    v1 = np.linspace(*simulation['mesh_vector1'])
    v2 = np.linspace(*simulation['mesh_vector2'])
    v3 = np.linspace(*simulation['mesh_vector3'])

    angles = list()
    ids = list()
    dirs = list()

    for array_id, dir, a_start, a_stop, a_step in rotations:
        ids.append(array_id)
        dirs.append(dir)
        angles.append(np.arange(a_start, a_stop + a_step, a_step))

    # rules = [(id, dir, a) for id, dir, a in zip(ids, dirs, angles) for angles in zip(*angles)]
    zip_args = list()
    for id, dir, ang in zip(ids, dirs, angles):
        zip_args.append(zip(repeat(id), repeat(dir), ang))
    rotation_rules = list(zip(*zip_args))

    # check for existing file
    if os.path.isfile(file):

        # con = sql.connect(file)
        response = input('File ' + str(file) + ' already exists.\n' +
                         'Continue (c), overwrite (o), or do nothing (any other key)?')

        if response.lower() in ['o', 'overwrite']:

            os.remove(file)

            # determine frequencies and wavenumbers
            field_pos = sim.meshview(v1, v2, v3, mode=mode)

            # create database
            with closing(sql.connect(file)) as con:

                # create database tables
                sim.create_metadata_table(con, **args, **simulation)
                create_field_positions_table(con, field_pos)
                create_pressures_table(con)

        elif response.lower() in ['c', 'continue']:

            with closing(sql.connect(file)) as con:
                table = pd.read_sql('SELECT x, y, z FROM field_position WHERE is_complete=0', con)
            field_pos = np.atleast_2d(np.array(table))

        else:
            raise Exception('Database already exists')

    else:

        # Make directories if they do not exist
        file_dir = os.path.dirname(os.path.abspath(file))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # determine frequencies and wavenumbers
        field_pos = sim.meshview(v1, v2, v3, mode=mode)

        # create database
        with closing(sql.connect(file)) as con:

            # create database tables
            sim.create_metadata_table(con, **args, **simulation)
            create_field_positions_table(con, field_pos)
            create_pressures_table(con)

    try:

        # start multiprocessing pool and run process
        write_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock,))

        simulation = abstract.dumps(simulation)
        arrays = abstract.dumps(arrays)
        proc_args = [(file, fp, rule, simulation, arrays) for fp in sim.chunks(field_pos, 100)
                     for rule in rotation_rules]
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Simulating', total=len(proc_args)):
            pass

        pool.close()

    except Exception as e:

        print(e)
        pool.terminate()
        pool.close()


## DATABASE FUNCTIONS ##

# Tables: metadata, field_positions, pressures, progress, image_data

def create_field_positions_table(con, field_pos):

    x, y, z = field_pos.T

    with con:
        # create table
        con.execute('CREATE TABLE field_positions (x float, y float, z float)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX field_position_index ON field_positions (x, y, z)')
        # insert values into table
        con.executemany('INSERT INTO field_positions VALUES (?, ?, ?)', zip(x, y, z, repeat(False)))


def create_rules_table(con, rotation_rules):

    with con:
        # create table
        con.execute('CREATE TABLE rules (id INTEGER KEY)')

        for array_no in range(len(rotation_rules[0])):

            con.execute('ALTER TABLE rules ADD COLUMN array_id_' + str(array_no) + ' int')
            con.execute('ALTER TABLE rules ADD COLUMN direction_' + str(array_no) + ' string')
            con.execute('ALTER TABLE rules ADD COLUMN angle_' + str(array_no) + ' float')

        for rule_no, rule in enumerate(rotation_rules):

            con.execute('INSERT INTO rules (id) VALUES (?)', [rule_no,])

            for array_no, (array_id, dir, angle) in enumerate(rule):
                query = '''
                        UPDATE rules SET array_id_%d = ?, direction_%d = ?, angle_%d = ?
                        WHERE id = ?
                        ''' % (array_no, array_no, array_no)
                con.execute(query, [array_id, dir, angle, rule_no])


def create_pressures_table(con):

    with con:
        # create table
        query = '''
                CREATE TABLE pressures (
                id INTEGER PRIMARY KEY,
                rule_id int,
                x float,
                y float,
                z float,
                time float,
                pressure float,
                FOREIGN KEY (x, y, z) REFERENCES field_positions (x, y, z),
                FOREIGN KEY (rule_id) REFERENCES rules (id)
                )
                '''
        con.execute(query)
        # create indexes
        con.execute('CREATE INDEX pressure_index ON pressures (x, y, z, time)')


def create_image_table(con):

    with con:
        # create table
        query = '''
                CREATE TABLE image (
                id INTEGER PRIMARY KEY,
                rule_id int,
                x float,
                y float,
                z float,
                brightness float,
                FOREIGN KEY (x, y, z) REFERENCES field_positions (x, y, z),
                FOREIGN KEY (rule_id) REFERENCES rules (id)
                )
                '''
        con.execute(query)
        # create indexes
        con.execute('CREATE INDEX pressure_index ON pressures (x, y, z, time)')


def update_progress(con, field_pos):

    x, y, z = np.array(field_pos).T

    with con:
        con.executemany('UPDATE field_positions SET is_complete=1 WHERE x=? AND y=? AND z=?', zip(x, y, z))


def update_pressures_table(con, field_pos, times, pressures):

    x, y, z = np.array(field_pos).T

    with con:
        query = '''
                INSERT INTO pressures (x, y, z, time, pressure) 
                VALUES (?, ?, ?, ?, ?)
                '''
        con.executemany(query, zip(x, y, z, times, pressures.ravel()))


## COMMAND LINE INTERFACE ##

def parse_rotation(array_id, direction, angle_start, angle_stop, angle_step):
    return int(array_id), direction, float(angle_start), float(angle_stop), float(angle_step)


if __name__ == '__main__':

    import argparse

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?')
    parser.add_argument('-s', '--specification', nargs='+')
    parser.add_argument('-t', '--threads', type=int)
    parser.add_argument('-r', '--rotation', nargs=5, action='append')
    parser.set_defaults(**defaults)

    args = vars(parser.parse_args())
    main(**args)



