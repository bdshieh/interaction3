
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
import pandas as pd
from itertools import repeat
from contextlib import closing
from tqdm import tqdm

from interaction3.bem.simulations.array_transmit_simulation import ArrayTransmitSimulation, connector
from interaction3 import abstract

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)


## PROCESS FUNCTIONS ##

def process(args):

    file, f, k, simulation, array = args

    simulation = ArrayTransmitSimulation(connector([simulation, array]))
    simulation.solve()

    nodes = simulation.result['nodes']
    membrane_ids = simulation.result['membrane_id']
    channel_ids = simulation.result['channel_id']
    x = simulation.result['x']

    with closing(sql.connect(file)) as conn:
        update_displacements_table(conn, f, k, nodes, membrane_ids, channel_ids, x)
        update_progress(conn, f)


## DATABASE FUNCTIONS ##

def create_metadata_table(conn, **kwargs):
    '''
    '''
    table = [[str(v) for v in list(kwargs.values())]]
    columns = list(kwargs.keys())

    pd.DataFrame(table, columns=columns, dtype=str).to_sql('metadata', conn, if_exists='replace', index=False)


def create_frequencies_table(conn, fs, ks):
    '''
    '''
    # create table
    query = '''
            CREATE TABLE frequencies ( 
            frequency float,
            wavenumber float,
            is_complete boolean
            )
            '''
    conn.execute(query)

    # create unique index on frequency
    query = '''
            CREATE UNIQUE INDEX frequency_index ON frequencies (frequency)
            '''
    conn.execute(query)

    # create unique index on wavenumber
    query = '''
            CREATE UNIQUE INDEX wavenumber_index ON frequencies (wavenumber)
            '''
    conn.execute(query)

    # insert values into table
    query = '''
            INSERT INTO frequencies (frequency, wavenumber, is_complete)
            VALUES (?, ?, ?)
            '''
    conn.executemany(query, zip(fs, ks, repeat(False)))

    conn.commit()


def update_progress(conn, f):

    query = '''
            UPDATE frequencies SET is_complete=1 WHERE frequency=?
            '''
    conn.execute(query, [f,])

    conn.commit()


# def create_nodes_table(conn, nodes):
#
#     x, y, z = nodes.T
#
#     query = '''
#             CREATE TABLE nodes (x float, y float, z float)
#             '''
#     conn.execute(query)
#
#     query = '''
#             CREATE UNIQUE INDEX node_index ON nodes (x, y, z)
#             '''
#     conn.execute(query)
#
#     query = '''
#             INSERT INTO nodes VALUES (?, ?, ?)
#             '''
#     conn.executemany(query, zip(x, y, z))
#
#     conn.commit()


def create_displacements_table(conn):

    query = '''
            CREATE TABLE displacements (
            id INTEGER PRIMARY KEY,
            frequency float,
            wavenumber float,
            x float,
            y float,
            z float,
            membrane_id int,
            channel_id int,
            displacement_real float,
            displacement_imag float,
            FOREIGN KEY (frequency) REFERENCES frequencies (frequency),
            FOREIGN KEY (wavenumber) REFERENCES frequencies (wavenumber)
            )
            '''
    conn.execute(query)

    query = '''
            CREATE INDEX translation_index ON displacements (frequency, x, y, z)
            '''
    conn.execute(query)

    query = '''
            CREATE INDEX translation_index ON displacements (membrane_id)
            '''
    conn.execute(query)

    query = '''
            CREATE INDEX translation_index ON displacements (channel_id)
            '''
    conn.execute(query)

    conn.commit()


def update_displacements_table(conn, f, k, nodes, membrane_ids, channel_ids, displacements):

    x, y, z = nodes.T

    query = '''
            INSERT INTO translations
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
    conn.executemany(query, zip(repeat(f), repeat(k), repeat(x), repeat(y), repeat(z), membrane_ids, channel_ids,
                                np.real(displacements.ravel()), np.imag(displacements.ravel())))

    conn.commit()


## MAIN FUNCTIONS ##

def read_json_files(*args):

    spec = list()

    for arg in args:

        obj = abstract.load(arg)

        if isinstance(obj, list):
            spec += obj
        else:
            spec.append(obj)

    return spec


def get_objects_from_spec(spec):

    for obj in spec:

        if isinstance(obj, abstract.Array):
            array = obj

        elif isinstance(obj, abstract.Simulation):
            simulation = obj

    return simulation, array


## ENTRY POINT ##

def main(**kwargs):


    threads = kwargs['threads']
    freqs = kwargs['freqs']
    file = kwargs['file']
    spec = kwargs['specification']

    simulation, array = get_objects_from_spec(read_json_files(spec))

    # set default threads to logical core count
    if threads is None:

        if 'threads' in simulation:
            threads = simulation['threads']
        else:
            threads = multiprocessing.cpu_count()

        kwargs['threads'] = threads

    if freqs is None:

        if 'freqs' in simulation:
            freqs = simulation['freqs']
        else:
            freqs = 50e3, 50e6, 50e3

        kwargs['freqs'] = freqs

    c = simulation['sound_speed']

    # check for existing file
    if os.path.isfile(file):

        # conn = sql.connect(file)
        response = input('File ' + str(file) + ' already exists. \nContinue (c), Overwrite (o), or Do nothing ('
                                                   'any other key)?')

        if response.lower() in ['o', 'overwrite']:

            os.remove(file)

            # determine frequencies and wavenumbers
            f_start, f_stop, f_step = freqs
            fs = np.arange(f_start, f_stop + f_step, f_step)
            ks = 2 * np.pi * fs / c

            # create database
            with closing(sql.connect(file)) as conn:

                # create database tables
                create_metadata_table(conn, **kwargs)
                create_frequencies_table(conn, fs, ks)
                create_displacements_table(conn)

        elif response.lower() in ['c', 'continue']:

            with closing(sql.connect(file)) as conn:

                query = '''
                        SELECT (frequency, wavenumber) FROM frequencies 
                        WHERE is_complete=False
                        '''
                table = pd.read_sql(query, conn)

            fs = np.array(table['frequency'])
            ks = np.array(table['wavenumber'])

        else:
            raise Exception('Database already exists')

    else:

        # Make directories if they do not exist
        file_dir = os.path.dirname(file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # determine frequencies and wavenumbers
        f_start, f_stop, f_step = freqs
        fs = np.arange(f_start, f_stop + f_step, f_step)
        ks = 2 * np.pi * fs / c

        # create database
        with closing(sql.connect(file)) as conn:

            # create database tables
            create_metadata_table(conn, **kwargs)
            create_frequencies_table(conn, fs, ks)
            create_displacements_table(conn)

    try:

        # start multiprocessing pool and run process
        pool = multiprocessing.Pool(threads)
        proc_args = [(file, f, k, simulation, array) for f, k in zip(fs, ks)]
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Simulating', total=len(fs)):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    # command line ideas ...
    # interaction3 abstract matrix_array myarray.json --options ...
    # interaction3 abstract simulation mysim.json --options ...
    # interaction3 bem array_transmit_simulation myoutput.db -i myarray.json mysim.json ...
    # interaction3 bem array_transmit_simulation myoutput.db -i myspecification.json

    import argparse

    # default arguments
    nthreads = None
    freqs = None
    file = None
    specification = None

    # max_level = 6
    # dims = 4e-3, 4e-3
    # sound_speed = 1500
    # density = 1000
    # orders_db = None
    # translations_db = None
    # bbox = [0, 1, 0, 1]
    # max_iterations = 100
    # tolerance = 0.01
    # use_preconditioner = True
    # use_pressure_load = False

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=file)
    parser.add_argument('-s', '--specification', nargs='+', default=specification)
    parser.add_argument('-t', '--threads', nargs=1, type=int, default=nthreads)
    parser.add_argument('-f', '--freqs', nargs=3, type=float, default=freqs)

    args = vars(parser.parse_args())
    main(**args)



