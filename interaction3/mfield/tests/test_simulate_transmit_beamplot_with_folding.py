## mfield / tests/ test_simulate_transmit_beamplot.py

import numpy as np
import subprocess

from interaction3 import abstract
from interaction3.abstract.arrays import foldable_linear_array
from interaction3.abstract import MfieldSimulation as Simulation

arrays = abstract.arrays.foldable_linear_array.init()

sim = Simulation(transmit_focus=[0, 0, 0.05],
                 delay_quantization=0,
                 threads=4,
                 rotations=[[0, 'y'], [2, '-y']],
                 angles=[0, 1, 1],
                 sampling_frequency=100e6,
                 sound_speed=1500,
                 use_attenuation=True,
                 frequency_attenuation=0,
                 attenuation_center_frequency=1e6,
                 excitation_center_frequecy=7e6,
                 excitation_bandwidth=5.6e6,
                 use_element_factor=False,
                 element_factor_file='',
                 mesh_mode='cartesian',
                 mesh_vector1=[0, 0.05, 51],
                 mesh_vector2=[0, 0.05, 51],
                 mesh_vector3=[0.05, 0.06, 1],
                 save_image_data_only=True
                 )

abstract.dump((sim,) + arrays, 'spec.json', mode='w')

# command = '''
#           python -m interaction3.mfield.scripts.simulate_transmit_receive_beamplot_with_folding_error
#           test.db
#           -s spec.json
#           '''
# subprocess.run(command.split())
#
# import sqlite3 as sql
# import pandas as pd
# from matplotlib import pyplot as plt
#
# con = sql.connect('test.db')
# image = np.array(pd.read_sql('SELECT brightness FROM image WHERE angle=0 ORDER BY x, y, z', con))
# plt.imshow(20*np.log10(np.abs(image)).reshape((41,41)).T)
# plt.show()