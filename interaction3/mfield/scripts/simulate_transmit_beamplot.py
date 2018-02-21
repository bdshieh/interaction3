
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
from interaction3.mfield.simulations import sim_functions as sim

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)


## PROCESS FUNCTIONS ##

def init_process(_write_lock, sim, array):

    global write_lock, simulation
    write_lock = _write_lock

    sim = abstract.loads(sim)
    array = abstract.loads(array)

    kwargs, meta = TransmitBeamplot.connector(sim, array)
    simulation = TransmitBeamplot(**kwargs)


def process(args):

    file, field_pos = args

    simulation.solve(field_pos)

    rf_data = simulation.result['rf_data']
    times = simulation.result['times']
    # t0s = simulation.result['t0s']

    with write_lock:
        with closing(sql.connect(file)) as con:

            for rf, t in zip(rf_data, times):
                update_pressures_table(con, field_pos, t, rf)
            update_progress(con, field_pos)


## ENTRY POINT ##

def main(**args):

    threads = args['threads']
    file = args['file']
    spec = args['specification']

    simulation, array = TransmitBeamplot.get_objects_from_spec(*spec)

    mode = simulation['mesh_mode']
    v1 = np.linspace(*simulation['mesh_vector1'])
    v2 = np.linspace(*simulation['mesh_vector2'])
    v3 = np.linspace(*simulation['mesh_vector3'])

    # set arg defaults if not provided
    if 'threads' in simulation:
        threads = simulation.pop('threads')
        args['threads'] = threads

    # check for existing file
    if os.path.isfile(file):

        # con = sql.connect(file)
        response = input('File ' + str(file) + ' already exists. \nContinue (c), Overwrite (o), or Do nothing ('
                                                   'any other key)?')

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

                query = 'SELECT x, y, z FROM field_position WHERE is_complete=0'
                table = pd.read_sql(query, con)

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
        simulation = abstract.dumps(simulation)
        array = abstract.dumps(array)

        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock, simulation, array))
        proc_args = [(file, fp) for fp in sim.chunks(field_pos, 100)]
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Simulating', total=len(proc_args)):
            pass

        pool.close()

    except Exception as e:

        print(e)
        pool.terminate()
        pool.close()





## DATABASE FUNCTIONS ##

# Tables: metadata, field_positions, pressures

def create_field_positions_table(con, field_pos):

    x, y, z = field_pos.T

    with con:

        # create table
        con.execute('CREATE TABLE field_positions (x float, y float, z float, is_complete boolean)')

        # create indexes
        con.execute('CREATE UNIQUE INDEX field_position_index ON field_positions (x, y, z)')

        # insert values into table
        query = 'INSERT INTO field_positions VALUES (?, ?, ?, ?)'
        con.executemany(query, zip(x, y, z, repeat(False)))


def create_pressures_table(con):

    with con:

        # create table
        query = '''
                CREATE TABLE pressures (
                id INTEGER PRIMARY KEY,
                x float,
                y float,
                z float,
                time float,
                pressure float,
                FOREIGN KEY (x, y, z) REFERENCES nodes (x, y, z),
                FOREIGN KEY (time) REFERENCES times (time)
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

if __name__ == '__main__':

    # command line ideas ...

    import argparse

    # default arguments
    nthreads = multiprocessing.cpu_count()
    file = None
    specification = None

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=file)
    parser.add_argument('-s', '--specification', nargs='+', default=specification)
    parser.add_argument('-t', '--threads', type=int, default=nthreads)

    args = vars(parser.parse_args())
    main(**args)



