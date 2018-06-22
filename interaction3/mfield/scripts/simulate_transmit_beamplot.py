## interaction3 / mfield / scripts / simulate_transmit_beamplot.py

import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from contextlib import closing
from tqdm import tqdm
import traceback
import sys

import interaction3.abstract as abstract
from interaction3.mfield.simulations import TransmitBeamplot
from interaction3.mfield.simulations import sim_functions as sim

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

defaults = dict()
defaults['threads'] = multiprocessing.cpu_count()


## PROCESS FUNCTIONS ##

def init_process(_write_lock, _simulation, _array):

    global write_lock, simulation, array

    write_lock = _write_lock
    simulation = abstract.loads(_simulation)
    array = abstract.loads(_array)


def process(job):

    job_id, (file, field_pos) = job

    kwargs, meta = TransmitBeamplot.connector(simulation, array)
    solver = TransmitBeamplot(**kwargs)

    rf_data = solver.result['rf_data']
    p = np.max(sim.envelope(rf_data, axis=1), axis=1)

    with write_lock:
        with closing(sql.connect(file)) as con:
            update_pressures_table(con, field_pos, p)
            sim.update_progress(con, job_id)


def run_process(*args, **kwargs):

    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


## ENTRY POINT ##

def main(**args):

    # get abstract objects from specification
    spec = args['spec']

    simulation, array = TransmitBeamplot.get_objects_from_spec(*spec)

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

    # create field positions
    v1 = np.linspace(*simulation['mesh_vector1'])
    v2 = np.linspace(*simulation['mesh_vector2'])
    v3 = np.linspace(*simulation['mesh_vector3'])
    field_pos = sim.meshview(v1, v2, v3, mode=mode)

    is_complete = None
    njobs = int(np.ceil(len(field_pos) / 100))
    ijob = 0

    # check for existing file
    if os.path.isfile(file):

        response = input('File ' + str(file) + ' already exists.\n' +
                         'Continue (c), overwrite (o), or do nothing (any other key)?')

        if response.lower() in ['o', 'overwrite']:

            os.remove(file)

            # create database
            with closing(sql.connect(file)) as con:

                # create database tables
                sim.create_metadata_table(con, **args)
                sim.create_progress_table(con, njobs)
                create_field_positions_table(con, field_pos)
                create_pressures_table(con)

        elif response.lower() in ['c', 'continue']:
            is_complete, ijob = sim.get_progress(file)

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
            sim.create_metadata_table(con, **args)
            sim.create_progress_table(con, njobs)
            create_field_positions_table(con, field_pos)
            create_pressures_table(con)

    try:

        # start multiprocessing pool and run process
        write_lock = multiprocessing.Lock()
        simulation = abstract.dumps(simulation)
        array = abstract.dumps(array)
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock, simulation, array))

        jobs = sim.create_jobs(file, (field_pos, 100), mode='zip', is_complete=is_complete)
        result = pool.imap_unordered(process, jobs)

        for r in tqdm(result, desc='Simulating', total=njobs, initial=ijob):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()



## DATABASE FUNCTIONS ##

# Tables: metadata, field_positions, pressures

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


def create_pressures_table(con):

    with con:

        # create table
        query = '''
                CREATE TABLE pressures (
                id INTEGER PRIMARY KEY,
                x float,
                y float,
                z float,
                pressure float,
                FOREIGN KEY (x, y, z) REFERENCES nodes (x, y, z)
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX pressure_index ON pressures (x, y, z)')


def update_pressures_table(con, field_pos, pressures):

    x, y, z = np.array(field_pos).T

    with con:
        query = 'INSERT INTO pressures (x, y, z,  pressure) VALUES (?, ?, ?, ?)'
        con.executemany(query, zip(x, y, z, pressures.ravel()))


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    import argparse

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?')
    parser.add_argument('-s', '--spec', nargs='+')
    parser.add_argument('-t', '--threads', type=int)

    args = vars(parser.parse_args())
    main(**args)

