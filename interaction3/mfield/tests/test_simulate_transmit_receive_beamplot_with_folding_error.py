## mfield / tests/ test_simulate_transmit_beamplot.py

import numpy as np
import subprocess

from interaction3 import abstract
from interaction3.abstract.arrays import foldable_linear_array
from interaction3.abstract import MfieldTransmitReceiveBeamplotWithFoldingError as Simulation

arrays = abstract.arrays.foldable_linear_array.init()

sim = Simulation(transmit_focus=[0, 0, 0.05],
                 receive_focus=[0, 0, 0.05],
                 delay_quantization=0,
                 threads=4,
                 rotations=[[0, 'y'], [2, '-y']],
                 angles=[0, 5, 1],
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
                 mesh_vector1=[-0.01, 0.01, 51],
                 mesh_vector2=[-0.01, 0.01, 51],
                 mesh_vector3=[0.05, 0.06, 1],
                 save_image_data_only=False
                 )

abstract.dump(arrays + (sim,), 'spec.json', mode='w')

command = '''
          python -m interaction3.mfield.scripts.simulate_transmit_receive_beamplot_with_folding_error 
          test.db
          -s spec.json
          '''
subprocess.run(command.split())
