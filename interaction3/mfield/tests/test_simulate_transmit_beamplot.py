## mfield / tests/ test_simulate_transmit_beamplot.py

import numpy as np
import subprocess

from interaction3 import abstract
from interaction3.abstract.arrays import matrix_array
from interaction3.mfield.simulations import TransmitBeamplot


array = abstract.arrays.matrix_array.init(nelem=[2,2])

simulation = abstract.MfieldTransmitBeamplot(sampling_frequency=100e6,
                                             sound_speed=1500,
                                             use_attenuation=True,
                                             frequency_attenuation=0,
                                             attenuation_center_frequency=1e6,
                                             excitation_center_frequecy=5e6,
                                             excitation_bandwidth=4e6,
                                             use_element_factor=False,
                                             element_factor_file='',
                                             mesh_mode='sector',
                                             mesh_vector1=[0.05, 0.06, 1],
                                             mesh_vector2=[np.deg2rad(-10), np.deg2rad(10), 21],
                                             mesh_vector3=[np.deg2rad(-10), np.deg2rad(10), 21]
                                             )

abstract.dump([array, simulation], 'test_spec.json', mode='w')

subprocess.run(['python', '-m', 'mfield.scripts.simulate_transmit_beamplot', 'test.db', '-t', '4', '-s',
                'test_spec.json'])
