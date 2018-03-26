
import os
import subprocess

from interaction3 import abstract
import interaction3.abstract.arrays.matrix_array
from interaction3.bem.simulations import TransmitCrosstalk

array = abstract.arrays.matrix_array.init(nelem=[2,2])
simulation = abstract.BemSimulation(freqs=[50e3, 5e6, 50e3],
                                    plane_wave_vector=[0, 0, -1],
                                    plane_wave_pressure=1,
                                    density=1000,
                                    sound_speed=1500,
                                    max_level=5,
                                    bounding_box=[-1e-3, -1e-3, 1e-3, 1e-3],
                                    orders_db='./orders_dims_0.0020_0.0020.db',
                                    translations_db='./translations_dims_0.0020_0.0020.db')

abstract.dump((simulation, array), 'test_spec.json', mode='w')
file = 'test.db'
if os.path.isfile(file):
    os.remove(file)

command = '''
          python -m interaction3.bem.scripts.simulate_receive_crosstalk
          test.db
          -s test_spec.json
          '''
subprocess.run(command.split())