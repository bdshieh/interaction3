
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
import pandas as pd
from itertools import repeat
from contextlib import closing
from tqdm import tqdm

from interaction3.bem.simulations.array_transmit_simulation import ArrayTransmitSimulation
from interaction3.bem.simulations.array_transmit_simulation import connector, get_objects_from_spec
from interaction3 import abstract

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)


## PROCESS FUNCTIONS ##

def process(args):

    file, f, k, sim, array = args

    sim['frequency'] = f
    kwargs, meta = connector(sim, array)

    simulation= ArrayTransmitSimulation(**kwargs)
    simulation.solve()

    nodes = simulation.nodes
    membrane_ids = meta['membrane_id']
    element_ids = meta['element_id']
    channel_ids = meta['channel_id']
    displacement = simulation.result['displacement']

    with closing(sql.connect(file)) as con:

        if not table_exists(con, 'nodes'):
            create_nodes_table(con, nodes, membrane_ids, element_ids, channel_ids)

        if not table_exists(con, 'displacements'):
            create_displacements_table(con)

        update_displacements_table(con, f, k, nodes, displacement)
        update_progress(con, f)


## ENTRY POINT ##

def main(**args):

    threads = args['threads']
    freqs = args['freqs']
    file = args['file']
    spec = args['specification']

    simulation, array = get_objects_from_spec(*spec)

    # set arg defaults if not provided
    if 'threads' in simulation:
        threads = simulation.pop('threads')
        args['threads'] = threads

    if 'freqs' in simulation:
        freqs = simulation.pop('freqs')
        args['freqs'] = freqs

    c = simulation['sound_speed']

    # check for existing file
    if os.path.isfile(file):

        # con = sql.connect(file)
        response = input('File ' + str(file) + ' already exists. \nContinue (c), Overwrite (o), or Do nothing ('
                                                   'any other key)?')

        if response.lower() in ['o', 'overwrite']:

            os.remove(file)

            # determine frequencies and wavenumbers
            f_start, f_stop, f_step = freqs
            fs = np.arange(f_start, f_stop + f_step, f_step)
            ks = 2 * np.pi * fs / c

            # create database
            with closing(sql.connect(file)) as con:

                # create database tables
                create_metadata_table(con, **args, **simulation)
                create_frequencies_table(con, fs, ks)

        elif response.lower() in ['c', 'continue']:

            with closing(sql.connect(file)) as con:

                query = '''
                        SELECT (frequency, wavenumber) FROM frequencies WHERE is_complete=False
                        '''
                table = pd.read_sql(query, con)

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
        with closing(sql.connect(file)) as con:

            # create database tables
            create_metadata_table(con, **args)
            create_frequencies_table(con, fs, ks)

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


## DATABASE FUNCTIONS ##

def create_metadata_table(con, **kwargs):

    table = [[str(v) for v in list(kwargs.values())]]
    columns = list(kwargs.keys())

    pd.DataFrame(table, columns=columns, dtype=str).to_sql('metadata', con, if_exists='replace', index=False)


def create_frequencies_table(con, fs, ks):

    with con:

        # create table
        con.execute('CREATE TABLE frequencies (frequency float, wavenumber float, is_complete boolean)')

        # create indexes
        con.execute('CREATE UNIQUE INDEX frequency_index ON frequencies (frequency)')
        con.execute('CREATE UNIQUE INDEX wavenumber_index ON frequencies (wavenumber)')

        # insert values into table
        con.executemany('INSERT INTO frequencies VALUES (?, ?, ?)', zip(fs, ks, repeat(False)))


def update_progress(con, f):

    with con:
        con.execute('UPDATE frequencies SET is_complete=1 WHERE frequency=?', [f,])


def table_exists(con, name):

    query = '''SELECT count(*) FROM sqlite_master WHERE type='table' and name=?'''
    return con.execute(query, name).fetchone()[0] != 0

def create_nodes_table(con, nodes, membrane_ids, element_ids, channel_ids):

    x, y, z = nodes.T

    with con:

        # create table
        con.execute('CREATE TABLE nodes (x float, y float, z float, membrane_id, element_id, channel_id)')

        # create indexes
        con.execute('CREATE UNIQUE INDEX node_index ON nodes (x, y, z)')
        con.execute('CREATE INDEX membrane_id_index ON nodes (membrane_id)')
        con.execute('CREATE INDEX element_id_index ON nodes (element_id)')
        con.execute('CREATE INDEX channel_id_index ON channel_id_index (channel_id)')

        # insert values into table
        query = 'INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)'
        con.executemany(query, zip(x, y, z, membrane_ids, element_ids, channel_ids))


def create_displacements_table(con):

    with con:

        # create table
        query = '''
                CREATE TABLE displacements (
                id INTEGER PRIMARY KEY,
                frequency float,
                wavenumber float,
                x float,
                y float,
                z float,
                displacement_real float,
                displacement_imag float,
                FOREIGN KEY (frequency) REFERENCES frequencies (frequency),
                FOREIGN KEY (wavenumber) REFERENCES frequencies (wavenumber),
                FOREIGN KEY (x, y, z) REFERENCES nodes (x, y, z)
                )
                '''
        con.execute(query)

        # create indexes
        con.execute('CREATE INDEX query_index ON displacements (frequency, x, y, z)')


def update_displacements_table(con, f, k, nodes, displacements):

    x, y, z = nodes.T

    with con:
        query = 'INSERT INTO displacements VALUES (?, ?, ?, ?, ?, ?, ?)'
        con.executemany(query, zip(repeat(f), repeat(k), repeat(x), repeat(y), repeat(z),
                                    np.real(displacements.ravel()), np.imag(displacements.ravel())))


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    # command line ideas ...
    # interaction3 abstract matrix_array myarray.json --options ...
    # interaction3 abstract simulation mysim.json --options ...
    # interaction3 bem array_transmit_simulation myoutput.db -i myarray.json mysim.json ...
    # interaction3 bem array_transmit_simulation myoutput.db -i myspecification.json

    import argparse

    # default arguments
    nthreads = multiprocessing.cpu_count
    freqs = 50e3, 50e6, 50e3
    file = None
    specification = None

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=file)
    parser.add_argument('-s', '--specification', nargs='+', default=specification)
    parser.add_argument('-t', '--threads', type=int, default=nthreads)
    parser.add_argument('-f', '--freqs', nargs=3, type=float, default=freqs)

    args = vars(parser.parse_args())
    main(**args)



