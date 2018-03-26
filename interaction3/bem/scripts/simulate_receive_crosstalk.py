
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from itertools import repeat
from contextlib import closing
from tqdm import tqdm
import traceback
import sys

from interaction3.bem.simulations import ReceiveCrosstalk, sim_functions as sim
from interaction3 import abstract

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

defaults = dict()
defaults['threads'] = multiprocessing.cpu_count()


## PROCESS FUNCTIONS ##

def init_process(_write_lock):

    global write_lock
    write_lock = _write_lock


def process(job):

    job_id, (file, f, k, simulation, array) = job

    # remove enclosing lists
    f = f[0]
    k = k[0]

    # deserialize json objects
    simulation = abstract.loads(simulation)
    array = abstract.loads(array)

    simulation['frequency'] = f
    kwargs, meta = ReceiveCrosstalk.connector(simulation, array)

    simulation = ReceiveCrosstalk(**kwargs)
    simulation.solve()

    nodes = simulation.nodes
    membrane_ids = meta['membrane_id']
    element_ids = meta['element_id']
    channel_ids = meta['channel_id']
    displacement = simulation.result['displacement']

    with write_lock:
        with closing(sql.connect(file)) as con:

            if not sim.table_exists(con, 'nodes'):
                create_nodes_table(con, nodes, membrane_ids, element_ids, channel_ids)
            if not sim.table_exists(con, 'displacements'):
                create_displacements_table(con)

            update_displacements_table(con, f, k, nodes, displacement)
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

    simulation, array = ReceiveCrosstalk.get_objects_from_spec(*spec)

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
    c = args['sound_speed']
    file = args['file']
    threads = args['threads']
    f_start, f_stop, f_step = args['freqs']

    # create frequencies/wavenumbers
    fs = np.arange(f_start, f_stop + f_step, f_step)
    ks = 2 * np.pi * fs / c
    njobs = len(fs)
    is_complete = None
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
                create_frequencies_table(con, fs, ks)
                sim.create_progress_table(con, njobs)

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
            create_frequencies_table(con, fs, ks)
            sim.create_progress_table(con, njobs)

    try:

        # start multiprocessing pool and run process
        write_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock,))

        simulation = abstract.dumps(simulation)
        array = abstract.dumps(array)
        jobs = sim.create_jobs(file, (fs, 1), (ks, 1), simulation, array, mode='zip', is_complete=is_complete)
        result = pool.imap_unordered(process, jobs)

        for r in tqdm(result, desc='Simulating', total=njobs, initial=ijob):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


## DATABASE FUNCTIONS ##

def create_frequencies_table(con, fs, ks):
    with con:
        # create table
        con.execute('CREATE TABLE frequencies (frequency float, wavenumber float)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX frequency_index ON frequencies (frequency)')
        con.execute('CREATE UNIQUE INDEX wavenumber_index ON frequencies (wavenumber)')
        # insert values into table
        con.executemany('INSERT INTO frequencies VALUES (?, ?)', zip(fs, ks))


def create_nodes_table(con, nodes, membrane_ids, element_ids, channel_ids):
    x, y, z = nodes.T
    with con:
        # create table
        con.execute('CREATE TABLE nodes (x float, y float, z float, membrane_id, element_id, channel_id)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX node_index ON nodes (x, y, z)')
        con.execute('CREATE INDEX membrane_id_index ON nodes (membrane_id)')
        con.execute('CREATE INDEX element_id_index ON nodes (element_id)')
        con.execute('CREATE INDEX channel_id_index ON nodes (channel_id)')
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
        query = '''
                INSERT INTO displacements (frequency, wavenumber, x, y, z, displacement_real, displacement_imag) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
        con.executemany(query, zip(repeat(f), repeat(k), x, y, z, np.real(displacements.ravel()),
                                   np.imag(displacements.ravel())))


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    import argparse

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?')
    parser.add_argument('-s', '--spec', nargs='+')
    parser.add_argument('-t', '--threads', type=int)
    parser.add_argument('-f', '--freqs', nargs=3, type=float)

    args = vars(parser.parse_args())
    main(**args)



