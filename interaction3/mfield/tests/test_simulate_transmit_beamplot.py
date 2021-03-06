## interaction3 / mfield / tests / test_simulate_transmit_beamplot.py

import numpy as np
import subprocess

from interaction3 import abstract
from interaction3.arrays import foldable_vernier

array_kwargs = {}
array_kwargs['ntransmit'] = 10
array_kwargs['nreceive'] = 10

sim_kwargs = {}
sim_kwargs['transmit_focus'] = [0, 0, 0.05]
sim_kwargs['receive_focus'] = [0, 0, 0.05]
sim_kwargs['delay_quantization'] = 0
sim_kwargs['threads'] = 3
sim_kwargs['sampling_frequency'] = 100e6
sim_kwargs['sound_speed'] = 1540
sim_kwargs['excitation_center_frequecy'] = 7e6
sim_kwargs['excitation_bandwidth'] = 5.6e6
sim_kwargs['mesh_mode'] ='cartesian'
sim_kwargs['mesh_vector1'] = [-0.02, 0.02, 41]
sim_kwargs['mesh_vector2'] = [-0.02, 0.02, 41]
sim_kwargs['mesh_vector3'] = [0.05, 0.06, 1]

arrays = foldable_vernier.create(**array_kwargs)
simulation = abstract.MfieldSimulation(**sim_kwargs)

abstract.dump((simulation,) + arrays, 'test_spec.json', mode='w')

command = '''
          python -m interaction3.mfield.scripts.simulate_transmit_beamplot
          test_database.db
          -s test_spec.json
          '''
subprocess.run(command.split())

import sqlite3 as sql
import pandas as pd
from matplotlib import pyplot as plt

con = sql.connect('test_database.db')
image = np.array(pd.read_sql('SELECT brightness FROM image ORDER BY x, y, z', con))
image = image.reshape((41, 41))

plt.figure()
plt.imshow(20 * np.log10(np.abs(image) / image.max()))
plt.show()