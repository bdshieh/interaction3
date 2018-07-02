## interaction3 / mfield / scripts / simulate_transmit_receive_beamplot_with_folding_error.py

import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from itertools import repeat
from contextlib import closing
from tqdm import tqdm
import traceback
import sys

from interaction3 import abstract, util
from interaction3.mfield.solvers import TransmitReceiveBeamplot2 as Solver

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

# set default script parameters
defaults = {}
defaults['threads'] = multiprocessing.cpu_count()
defaults['focus'] = None
defaults['delay_quantization'] = 0


## PROCESS FUNCTIONS ##

POSITIONS_PER_PROCESS = 1000


def init_process(_write_lock, _simulation, _arrays):

    global write_lock, simulation, arrays_base

    write_lock = _write_lock
    simulation = abstract.loads(_simulation)
    arrays_base = abstract.loads(_arrays)

    # set focus delays while arrays are flat
    focus = simulation['focus']
    if focus is not None:
        c = simulation['sound_speed']
        delay_quant = simulation['delay_quantization']
        for array in arrays_base:
            abstract.focus_array(array, focus, c, delay_quant, kind='both')

    # important! avoid solver overriding current delays
    if 'transmit_focus' in simulation:
        simulation.pop('transmit_focus')
    if 'receive_focus' in simulation:
        simulation.pop('receive_focus')


def process(job):

    job_id, (file, field_pos, rotation_rule) = job

    rotation_rule = rotation_rule[0] # remove enclosing list

    arrays = arrays_base.copy()  # make copy of array so following manipulations do not persist

    # rotate arrays based on rotation rule
    for array_id, dir, angle in rotation_rule:
        for array in arrays:
            if array['id'] == array_id:
                abstract.rotate_array(array, dir, np.deg2rad(angle))

    # create and run simulation

    kwargs, meta = Solver.connector(simulation, *arrays)
    solver = Solver(**kwargs)
    solver.solve(field_pos)

    # extract results and save
    rf_data = solver.result['rf_data']
    p = np.max(util.envelope(rf_data, axis=1), axis=1)
    angle = rotation_rule[0][2]

    with write_lock:
        with closing(sql.connect(file)) as con:
            update_image_table(con, angle, field_pos, p)
            util.update_progress(con, job_id)


def run_process(*args, **kwargs):

    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


## ENTRY POINT ##

def main(**args):

    # get abstract objects from specification
    spec = args['spec']

    objects = Solver.get_objects_from_spec(*spec)
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

    print('simulation parameters as key --> value:')
    for k, v in args.items():
        print(k, '-->', v)

    # get args needed in main
    file = args['file']
    threads = args['threads']
    mode = args['mesh_mode']
    mesh_vector1 = args['mesh_vector1']
    mesh_vector2 = args['mesh_vector2']
    mesh_vector3 = args['mesh_vector3']
    rotations = args['rotations']
    a_start, a_stop, a_step = args['angles']

    # create field positions
    field_pos = util.meshview(np.linspace(*mesh_vector1), np.linspace(*mesh_vector2), np.linspace(*mesh_vector3),
                             mode=mode)

    # create angles
    angles = np.arange(a_start, a_stop + a_step, a_step)

    # create rotation rules which will be distributed by the pool
    array_ids = [id for id, _ in rotations]
    dirs = [dir for _, dir in rotations]
    zip_args = []
    for id, dir in zip(array_ids, dirs):
        zip_args.append(zip(repeat(id), repeat(dir), angles))
    rotation_rules = list(zip(*zip_args))

    # calculate job-related values
    is_complete = None
    njobs = int(np.ceil(len(field_pos) / POSITIONS_PER_PROCESS) * len(rotation_rules))
    ijob = 0

    # check for existing file
    if os.path.isfile(file):

        response = input('File ' + str(file) + ' already exists.\n' +
                         'Continue (c), overwrite (o), or do nothing (any other key)?')

        if response.lower() in ['o', 'overwrite']:  # if file exists, prompt for overwrite

            os.remove(file)  # remove existing file
            create_database(file, args, njobs, field_pos)  # create database

        elif response.lower() in ['c', 'continue']:  # continue from current progress
            is_complete, ijob = util.get_progress(file)

        else:
            raise Exception('Database already exists')

    else:

        # Make directories if they do not exist
        file_dir = os.path.dirname(os.path.abspath(file))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create database
        create_database(file, args, njobs, field_pos)

    try:

        # start multiprocessing pool and run process
        write_lock = multiprocessing.Lock()
        simulation = abstract.dumps(simulation)
        arrays = abstract.dumps(arrays)
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock, simulation, arrays))

        jobs = util.create_jobs(file, (field_pos, POSITIONS_PER_PROCESS), (rotation_rules, 1), mode='product',
                                is_complete=is_complete)
        result = pool.imap_unordered(run_process, jobs)

        for r in tqdm(result, desc='Simulating', total=njobs, initial=ijob):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


## DATABASE FUNCTIONS ##

def create_database(file, args, njobs, field_pos):

    with closing(sql.connect(file)) as con:
        # create database tables (metadata, progress, field_positions, pressures)
        util.create_metadata_table(con, **args)
        util.create_progress_table(con, njobs)
        create_field_positions_table(con, field_pos)
        create_image_table(con)


def create_field_positions_table(con, field_pos):

    x, y, z = np.atleast_2d(field_pos).T

    with con:
        # create table
        con.execute('CREATE TABLE field_positions (x float, y float, z float)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX field_position_index ON field_positions (x, y, z)')
        # insert values into table
        con.executemany('INSERT INTO field_positions VALUES (?, ?, ?)', zip(x, y, z))


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


def update_image_table(con, angle, field_pos, brightness):

    x, y, z = np.array(field_pos).T

    with con:
        query = 'INSERT INTO image (angle, x, y, z, brightness) VALUES (?, ?, ?, ?, ?)'
        con.executemany(query, zip(repeat(angle), x, y, z, brightness.ravel()))


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
    # parser.add_argument('-r', '--rotations', nargs=2, action='append', type=parse_rotation)
    # parser.add_argument('-a', '--angles', nargs=3, type=float)

    args = vars(parser.parse_args())
    main(**args)

