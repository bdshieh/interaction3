## interaction3 / mfield / scripts / simulate_transmit_beamplot.py

import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from contextlib import closing
from tqdm import tqdm
import traceback
import argparse
import sys

from interaction3 import abstract, util
from interaction3.mfield.solvers import TransmitBeamplot2 as TransmitBeamplot

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

# set default script parameters
defaults = {}
defaults['threads'] = multiprocessing.cpu_count()


## PROCESS FUNCTIONS ##

# lower number means positions are closer to each other,
# preventing excessive zero-padding when aligning time signals
POSITIONS_PER_PROCESS = 400


def init_process(_write_lock, _simulation, _arrays):

    global write_lock, solver

    write_lock = _write_lock
    simulation = abstract.loads(_simulation)
    arrays = abstract.loads(_arrays)

    kwargs, meta = TransmitBeamplot.connector(simulation, *arrays)
    solver = TransmitBeamplot(**kwargs)


def process(job):

    job_id, (file, field_pos) = job

    solver.solve(field_pos)

    # rf_data = solver.result['rf_data']
    # p = np.max(util.envelope(rf_data, axis=1), axis=1)
    p = solver.result['brightness']

    with write_lock:
        with closing(sql.connect(file)) as con:
            update_image_table(con, field_pos, p)
            util.update_progress(con, job_id)


def run_process(*args, **kwargs):

    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


## ENTRY POINT ##

def main(args):

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?')
    parser.add_argument('-s', '--spec', nargs='+')
    parser.add_argument('-t', '--threads', type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = vars(parser.parse_args(args))

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

    print('simulation parameters as key --> value:')
    for k, v in args.items():
        print(k, '-->', v)

    # get args needed in main
    file = args['file']
    threads = args['threads']
    overwrite = args['overwrite']
    mode = args['mesh_mode']
    mesh_vector1 = args['mesh_vector1']
    mesh_vector2 = args['mesh_vector2']
    mesh_vector3 = args['mesh_vector3']

    # create field positions
    field_pos = util.meshview(np.linspace(*mesh_vector1), np.linspace(*mesh_vector2), np.linspace(*mesh_vector3),
                             mode=mode)

    # calculate job-related values
    is_complete = None
    njobs = int(np.ceil(len(field_pos) / POSITIONS_PER_PROCESS))
    ijob = 0

    # check for existing file
    if os.path.isfile(file):

        if overwrite:  # if file exists, prompt for overwrite
            os.remove(file)  # remove existing file
            create_database(file, args, njobs, field_pos)  # create database

        else: # continue from current progress
            is_complete, ijob = util.get_progress(file)
            if np.all(is_complete):
                return

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

        jobs = util.create_jobs(file, (field_pos, POSITIONS_PER_PROCESS), mode='zip', is_complete=is_complete)
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

    x, y, z = field_pos.T

    with con:
        # create table
        con.execute('CREATE TABLE field_positions (x float, y float, z float)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX field_position_index ON field_positions (x, y, z)')
        # insert values into table
        query = 'INSERT INTO field_positions VALUES (?, ?, ?)'
        con.executemany(query, zip(x, y, z))


def create_image_table(con):

    with con:
        # create table
        query = '''
                CREATE TABLE image (
                id INTEGER PRIMARY KEY,
                x float,
                y float,
                z float,
                brightness float,
                FOREIGN KEY (x, y, z) REFERENCES nodes (x, y, z)
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX image_index ON image (x, y, z)')


def update_image_table(con, field_pos, brightness):

    x, y, z = np.array(field_pos).T

    with con:
        query = 'INSERT INTO image (x, y, z,  brightness) VALUES (?, ?, ?, ?)'
        con.executemany(query, zip(x, y, z, brightness.ravel()))


if __name__ == '__main__':
    main(sys.argv[1:])



