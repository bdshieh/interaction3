
import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from contextlib import closing
from tqdm import tqdm
import attr

from interaction3.bem.simulations.array_transmit_simulation import ArrayTransmitSimulation, connector

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)


## PROCESS FUNCTIONS ##

def process(args):

    json_file = args

    simulation = ArrayTransmitSimulation(connector(json_file))
    simulation.solve()

    write_to_db(simulation.result)


## DATABASE FUNCTIONS ##

def write_to_db():
    pass


## ENTRY POINT ##

def main(**kwargs):

    threads = kwargs['threads']
    freqs = kwargs['freqs']
    levels = kwargs['levels']
    dims = kwargs['dims']
    c = kwargs['sound_speed']
    file = kwargs['file']
    orders_db = kwargs['orders_db']

    # set default threads to logical core count
    if threads is None:
        threads = multiprocessing.cpu_count()
        kwargs['threads'] = threads

    # path to this module's directory
    module_dir = os.path.dirname(os.path.realpath(__file__))


    # check for existing file
    if os.path.isfile(file):

        # conn = sql.connect(file)
        response = input('Database ' + str(file) + ' already exists. \nContinue (c), Overwrite (o), or Do nothing ('
                                                   'any other key)?')

        if response.lower() in ['o', 'overwrite']:

            os.remove(file)

            # determine frequencies and wavenumbers
            f_start, f_stop, f_step = freqs
            fs = np.arange(f_start, f_stop + f_step, f_step)
            ks = 2 * np.pi * fs / c

        elif response.lower() in ['c', 'continue']:

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

    try:

        # start multiprocessing pool and run process
        pool = multiprocessing.Pool(threads)
        proc_args = [(file, f, k, dims, levels, orders_db) for f, k in zip(fs, ks)]
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Building', total=len(fs)):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    import argparse

    # interaction3 abstract matrix_array myarray.json --options ...
    # interaction3 abstract simulation mysim.json --options ...
    # interaction3 bem array_transmit_simulation myoutput.db -i myarray.json mysim.json ...
    # interaction3 bem array_transmit_simulation myoutput.db -i myspecification.json
    pass


