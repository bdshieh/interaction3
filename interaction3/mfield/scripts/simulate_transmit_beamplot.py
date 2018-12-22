## interaction3 / mfield / scripts / simulate_transmit_beamplot.py

import numpy as np
import multiprocessing
import os
import sqlite3 as sql
from contextlib import closing
from tqdm import tqdm
import traceback
import sys

from interaction3 import abstract, util
from interaction3.mfield.solvers import TransmitBeamplot

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)


## PROCESS FUNCTIONS ##

def init_process(_write_lock, _cfg, _args):
    global write_lock, cfg, args
    write_lock = _write_lock
    cfg = _cfg
    args = _args


def process(job):

    job_id, (field_pos) = job

    solver_cfg = TransmitBeamplot.Config(**cfg._asdict())
    arrays = abstract.load(cfg.array_config)

    solver = TransmitBeamplot.from_abstract(solver_cfg, arrays)
    solver.solve(field_pos)

    rf_data = solver.result['rf_data']
    p = np.max(util.envelope(rf_data, axis=1), axis=1)

    with write_lock:
        update_pressures_table(args.file, field_pos, p)
        util.update_progress(args.file, job_id)


def run_process(*args, **kwargs):
    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


## DATABASE FUNCTIONS ##

def create_database(file, cfg, args, field_pos):
    with closing(sql.connect(file)) as con:
        # create database tables (metadata, progress, field_positions, pressures)
        util.create_metadata_table(con, **cfg._asdict(), **vars(args))
        create_field_positions_table(con, field_pos)
        create_pressures_table(con)


@util.open_db
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


@util.open_db
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


@util.open_db
def update_pressures_table(con, field_pos, pressures):
    x, y, z = np.array(field_pos).T
    with con:
        query = 'INSERT INTO pressures (x, y, z,  pressure) VALUES (?, ?, ?, ?)'
        con.executemany(query, zip(x, y, z, pressures.ravel()))


## ENTRY POINT ##

def main(cfg, args):

    # get parameters from config and args
    file = args.file
    write_over = args.write_over
    threads = args.threads if args.threads else multiprocessing.cpu_count()
    positions_per_process = cfg.positions_per_process

    # get args needed in main
    file = args.file
    threads = args.threads
    mode = cfg.mesh_mode
    mesh_vector1 = cfg.mesh_vector1
    mesh_vector2 = cfg.mesh_vector2
    mesh_vector3 = cfg.mesh_vector3

    # create field positions
    field_pos = util.meshview(np.linspace(*mesh_vector1), np.linspace(*mesh_vector2), np.linspace(*mesh_vector3),
                             mode=mode)

    # calculate job-related values
    is_complete = None
    njobs = int(np.ceil(len(field_pos) / positions_per_process))
    ijob = 0

    # check for existing file
    if os.path.isfile(file):
        if write_over:  # if file exists, write over
            os.remove(file)  # remove existing file
            create_database(file, cfg, args, field_pos)  # create database
            util.create_progress_table(file, njobs)

        else: # continue from current progress
            is_complete, ijob = util.get_progress(file)
            if np.all(is_complete): return
    else:
        # Make directories if they do not exist
        file_dir = os.path.dirname(os.path.abspath(file))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create database
        create_database(file, cfg, args, field_pos)  # create database
        util.create_progress_table(file, njobs)

    # start multiprocessing pool and run process
    try:
        write_lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock, cfg, args))
        jobs = util.create_jobs((field_pos, positions_per_process), mode='zip', is_complete=is_complete)
        result = pool.imap_unordered(run_process, jobs)
        for r in tqdm(result, desc='Calculating', total=njobs, initial=ijob):
            pass
    except Exception as e:
        print(e)
    finally:
        pool.terminate()
        pool.close()


if __name__ == '__main__':

    from interaction3 import util

    # define default configuration for this script
    Config = {}
    Config['array_config'] = ''
    Config['use_attenuation'] = False
    Config['attenuation'] = 0
    Config['frequency_attenuation'] = 0
    Config['attenuation_center_frequency'] = 1e6
    Config['use_element_factor'] = False
    Config['element_factor_file'] = None
    Config['field_positions'] = None
    Config['transmit_focus'] = [0, 0, 0.03]
    Config['sound_speed'] = 1500.
    Config['delay_quantization'] = 0
    Config['mesh_mode'] = 'sector'
    Config['mesh_vector1'] = -10, 10, 21
    Config['mesh_vector2'] = 0, 1, 1
    Config['mesh_vector3'] = 0, 1, 1
    Config['positions_per_process'] = 5000

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)

