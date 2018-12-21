## interaction3 / mfield / scripts / simulate_transmit_beamplot_with_corrected_folding_error.py

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
from interaction3.mfield.solvers import TransmitBeamplot

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)

# set default script parameters
defaults = dict()
defaults['threads'] = multiprocessing.cpu_count()
defaults['transmit_focus'] = None
defaults['delay_quantization'] = 0


## PROCESS FUNCTIONS ##

def init_process(_write_lock, _cfg, _args):

    global write_lock, cfg, args, arrays_base
    write_lock = _write_lock
    cfg = _cfg
    args = _args

    arrays_base = abstract.load(cfg.array_config)

    # set focus delays while arrays are flat
    tx_focus = cfg.transmit_focus
    if tx_focus is not None:
        c = cfg.sound_speed
        delay_quant = cfg.delay_quantization
        for array in arrays_base:
            abstract.focus_array(array, tx_focus, c, delay_quant, kind='tx')

    # simulation.pop('transmit_focus')  # important! avoid solver overriding current delays


def process(job):

    job_id, (field_pos, rotation_rule) = job

    arrays = arrays_base.copy()
    # rotate arrays based on rotation rule
    for array_id, dir, angle in rotation_rule:
        for array in arrays:
            if array.id == array_id:
                abstract.rotate_array(array, dir, np.deg2rad(angle))

    # create and run simulation
    solver = TransmitBeamplot.from_abstract(cfg, arrays)
    solver.solve(field_pos)

    # extract results and save
    rf_data = solver.result['rf_data']
    p = np.max(util.envelope(rf_data, axis=1), axis=1)
    angle = rotation_rule[0][2]

    with write_lock:
        update_pressures_table(args.file, angle, field_pos, p)
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
        util.create_metadata_table(con, **args)
        create_field_positions_table(con, field_pos)
        create_pressures_table(con)


@util.open_db
def create_field_positions_table(con, field_pos):
    x, y, z = np.atleast_2d(field_pos).T
    with con:
        # create table
        con.execute('CREATE TABLE field_positions (x float, y float, z float)')
        # create indexes
        con.execute('CREATE UNIQUE INDEX field_position_index ON field_positions (x, y, z)')
        # insert values into table
        con.executemany('INSERT INTO field_positions VALUES (?, ?, ?)', zip(x, y, z))


@util.open_db
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
                pressure float,
                FOREIGN KEY (x, y, z) REFERENCES field_positions (x, y, z)
                )
                '''
        con.execute(query)
        # create indexes
        con.execute('CREATE INDEX pressure_index ON pressures (angle, x, y, z)')


@util.open_db
def update_pressures_table(con, angle, field_pos, pressures):
    x, y, z = np.array(field_pos).T
    with con:
        query = 'INSERT INTO pressures (angle, x, y, z, pressure) VALUES (?, ?, ?, ?, ?)'
        con.executemany(query, zip(repeat(angle), x, y, z, pressures.ravel()))


## ENTRY POINT ##

def parse_rotation(string):
    try:
        return int(string)
    except ValueError:
        return string


def main(cfg, args):

    # get parameters from config and args
    file = args.file
    write_over = args.write_over
    threads = args.threads if args.threads else multiprocessing.cpu_count()
    positions_per_process = cfg.positions_per_process
    mode = cfg.mesh_mode
    mesh_vector1 = cfg.mesh_vector1
    mesh_vector2 = cfg.mesh_vector2
    mesh_vector3 = cfg.mesh_vector3
    rotations = cfg.rotations
    a_start, a_stop, a_step = cfg.angles

    # create field positions
    field_pos = util.meshview(np.linspace(*mesh_vector1), np.linspace(*mesh_vector2), np.linspace(*mesh_vector3),
                             mode=mode)
    
    # create rotation rules which will be distributed by the pool
    angles = np.arange(a_start, a_stop + a_step, a_step)
    array_ids = [id for id, _ in rotations]
    dirs = [dir for _, dir in rotations]
    zip_args = list()
    for id, dir in zip(array_ids, dirs):
        zip_args.append(zip(repeat(id), repeat(dir), angles))
    rotation_rules = list(zip(*zip_args))

    # calculate job-related values
    is_complete = None
    njobs = int(np.ceil(len(field_pos) / positions_per_process) * len(rotation_rules))
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
        jobs = util.create_jobs((field_pos, positions_per_process), (rotation_rules, 1), mode='product',
                                is_complete=is_complete)
        result = pool.imap_unordered(run_process, jobs)
        for r in tqdm(result, desc='Calculating', total=njobs, initial=ijob):
            pass
    except Exception as e:
        print(e)
    finally:
        pool.terminate()
        pool.close()


if __name__ == '__main__':

    # define default configuration for this script
    Config = {}
    Config['use_attenuation'] = False
    Config['attenuation'] = 0
    Config['frequency_attenuation'] = 0
    Config['attenuation_center_frequency'] = 1e6
    Config['use_element_factor'] = False
    Config['element_factor_file'] = None
    Config['field_positions'] = None
    Config['transmit_focus'] = None
    Config['sound_speed'] = 1500.
    Config['delay_quantization'] = 0
    Config['mesh_mode'] = 'sector'
    Config['mesh_vector1'] = None
    Config['mesh_vector2'] = None
    Config['mesh_vector3'] = None
    Config['rotations'] = None
    Config['angles'] = None
    Config['positions_per_process'] = 5000

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)

