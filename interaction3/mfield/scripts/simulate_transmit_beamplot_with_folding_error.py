
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
import pandas as pd
from itertools import repeat
from contextlib import closing
from tqdm import tqdm

import interaction3.abstract as abstract
from interaction3.mfield.simulations import TransmitBeamplot
from interaction3.mfield.simulations import sim_functions as simfuncs

import traceback
import sys
# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

defaults = dict()
defaults['threads'] = multiprocessing.cpu_count()
# defaults['spec'] = None
# defaults['file'] = None
# defaults['rotation'] = None
# defaults['angles'] = None


## PROCESS FUNCTIONS ##

def init_process(_write_lock):

    global write_lock
    write_lock = _write_lock


def process(args):

    job_id, (file, sim, arrays, field_pos, rotation_rule) = args

    rotation_rule = rotation_rule[0] # remove enclosing list

    sim = abstract.loads(sim)
    arrays = abstract.loads(arrays)

    # set focus delays while arrays are flat
    tx_focus = sim['transmit_focus']
    c = sim['sound_speed']
    quant = sim['delay_quantization']

    for array in arrays:
        abstract.focus_array(array, tx_focus, sound_speed=c, quantization=quant, kind='tx')

    # rotate arrays based on rotation rule
    for array_id, dir, angle in rotation_rule:
        for array in arrays:
            if array['id'] == array_id:
                abstract.rotate_array(array, dir, np.deg2rad(angle))

    # create and run simulation
    kwargs, meta = TransmitBeamplot.connector(sim, *arrays)
    simulation = TransmitBeamplot(**kwargs)
    simulation.solve(field_pos)

    # extract results and save
    rf_data = simulation.result['rf_data']
    times = simulation.result['times']
    angle = rotation_rule[0][2]

    with write_lock:
        with closing(sql.connect(file)) as con:

            for i, (rf, t) in enumerate(zip(rf_data, times)):

                # save image data
                b = np.max(simfuncs.envelope(rf))
                update_image_table(con, angle, field_pos[i], b)

                # save pressure data
                if not sim['save_image_data_only']:
                    update_pressures_table(con, angle, field_pos[i], t, rf)

            update_progress(con, job_id)


def run_process(*args, **kwargs):

    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


## ENTRY POINT ##

def main(**args):

    # get abstract objects from specification
    spec = args['spec']

    objects = TransmitBeamplot.get_objects_from_spec(*spec)
    simulation = objects[0]
    arrays = objects[1:]

    # set defaults with the following priority: command line arguments >> simulation object >> script defaults
    for k, v in simulation.items():
        args.setdefault(k, v)
        if args[k] is None:
            args[k] = v
    for k, v in defaults.items():
        args.setdefault(k, v)
        if args[k] is None:
            args[k] = v

    print(args)
    # get args needed in main
    threads = args['threads']
    file = args['file']
    rotations = args['rotations']
    a_start, a_stop, a_step = args['angles']

    # create field positions
    mode = simulation['mesh_mode']
    v1 = np.linspace(*simulation['mesh_vector1'])
    v2 = np.linspace(*simulation['mesh_vector2'])
    v3 = np.linspace(*simulation['mesh_vector3'])
    field_pos = simfuncs.meshview(v1, v2, v3, mode=mode)

    # create angles
    angles = np.arange(a_start, a_stop + a_step, a_step)

    # create rotation rules which will be distributed by the pool
    array_ids = [id for id, _ in rotations]
    dirs = [dir for _, dir in rotations]

    zip_args = list()
    for id, dir in zip(array_ids, dirs):
        zip_args.append(zip(repeat(id), repeat(dir), angles))
    rotation_rules = list(zip(*zip_args))

    # determine total number of jobs and initialize progress
    njobs = int(np.ceil(len(field_pos) / 100) * len(rotation_rules))
    is_complete = [False,] * njobs

    # check for existing file
    if os.path.isfile(file):

        # con = sql.connect(file)
        response = input('File ' + str(file) + ' already exists.\n' +
                         'Continue (c), overwrite (o), or do nothing (any other key)?')

        if response.lower() in ['o', 'overwrite']:

            os.remove(file)

            # create database
            with closing(sql.connect(file)) as con:

                # create database tables
                simfuncs.create_metadata_table(con, **args)
                create_field_positions_table(con, field_pos)
                create_progress_table(con, njobs)
                create_pressures_table(con)
                create_image_table(con)

        elif response.lower() in ['c', 'continue']:

            with closing(sql.connect(file)) as con:
                table = pd.read_sql('SELECT is_complete FROM progress SORT BY job_id', con)
            is_complete = np.array(table).squeeze()

        else:
            raise Exception('Database already exists')

    else:

        # Make directories if they do not exist
        file_dir = os.path.dirname(os.path.abspath(file))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create database
        with closing(sql.connect(file)) as con:

            # create database tables
            simfuncs.create_metadata_table(con, **args)
            create_field_positions_table(con, field_pos)
            create_progress_table(con, njobs)
            create_pressures_table(con)
            create_image_table(con)

    # try:

    # start multiprocessing pool and run process
    write_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock,))

    simulation = abstract.dumps(simulation)
    arrays = abstract.dumps(arrays)
    jobs = simfuncs.create_jobs(file, simulation, arrays, (field_pos, 100), (rotation_rules, 1), mode='product',
                           is_complete=is_complete)
    result = pool.imap_unordered(run_process, jobs)

    for r in tqdm(result, desc='Simulating', total=njobs):
        pass

    pool.close()

    # except Exception as e:
    #
    #     print(e)
    #     pool.terminate()
    #     pool.close()


## DATABASE FUNCTIONS ##

# Tables: metadata, field_positions, pressures, image_data, progress

def create_field_positions_table(con, field_pos):

    x, y, z = np.atleast_2d(field_pos).T

    with con:
        # create table
        con.execute('CREATE TABLE field_positions (x float, y float, z float)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX field_position_index ON field_positions (x, y, z)')
        # insert values into table
        con.executemany('INSERT INTO field_positions VALUES (?, ?, ?)', zip(x, y, z))


def create_progress_table(con, njobs):

    with con:
        # create table
        con.execute('CREATE TABLE progress (job_id INTEGER PRIMARY KEY, is_complete boolean)')
        # insert values
        con.executemany('INSERT INTO progress (is_complete) VALUES (?)', repeat((False,), njobs))


def create_pressures_table(con):

    with con:
        # create table
        query = '''
                CREATE TABLE pressures (
                id INTEGER PRIMARY KEY,
                angle float,
                x float,
                y float,
                z float,
                time float,
                pressure float,
                FOREIGN KEY (x, y, z) REFERENCES field_positions (x, y, z)
                )
                '''
        con.execute(query)
        # create indexes
        con.execute('CREATE INDEX pressure_index ON pressures (angle, x, y, z, time)')


def create_image_table(con):

    with con:
        # create table
        query = '''
                CREATE TABLE image (
                id INTEGER PRIMARY KEY,
                angle float,
                x float,
                y float,
                z float,
                brightness float,
                FOREIGN KEY (x, y, z) REFERENCES field_positions (x, y, z)
                )
                '''
        con.execute(query)
        # create indexes
        con.execute('CREATE INDEX image_index ON image (angle, x, y, z)')


def update_progress(con, job_id):

    with con:
        con.execute('UPDATE progress SET is_complete=1 WHERE job_id=?', [job_id,])


def update_pressures_table(con, angle, field_pos, times, pressures):

    x, y, z = field_pos

    with con:
        query = 'INSERT INTO pressures (angle, x, y, z, time, pressure) VALUES (?, ?, ?, ?, ?, ?)'
        con.executemany(query, zip(repeat(angle), repeat(x), repeat(y), repeat(z), times, pressures.ravel()))


def update_image_table(con, angle, field_pos, image_data):

    x, y, z = field_pos

    with con:
        query = 'INSERT INTO image (angle, x, y, z, brightness) VALUES (?, ?, ?, ?, ?)'
        con.executemany(query, zip(repeat(angle), repeat(x), repeat(y), repeat(z), image_data.ravel()))


## COMMAND LINE INTERFACE ##

def parse_rotation(string):
    try:
        return int(string)
    except ValueError:
        return string


if __name__ == '__main__':

    import argparse

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?')
    parser.add_argument('-s', '--spec', nargs='+')
    parser.add_argument('-t', '--threads', type=int)
    parser.add_argument('-r', '--rotations', nargs=2, action='append', type=parse_rotation)
    parser.add_argument('-a', '--angles', nargs=3, type=float)
    # parser.set_defaults(**defaults)

    args = vars(parser.parse_args())
    main(**args)

