
import numpy as np

from interaction3 import abstract, util
from interaction3.arrays import foldable_constant_spiral
from interaction3.mfield.solvers import TransmitBeamplot2 as TransmitBeamplot

field_positions = util.meshview(np.linspace(0.001, 0.061, 61),
                                np.linspace(-60, 60, 121),
                                np.linspace(0, 1, 1), mode='sector')


arrays = foldable_constant_spiral.create()

simulation = abstract.MfieldSimulation(sampling_frequency=100e6,
                                       sound_speed=1540,
                                       excitation_center_frequecy=5e6,
                                       excitation_bandwidth=4e6,
                                       field_positions=field_positions)


kwargs, meta = TransmitBeamplot.connector(simulation, *arrays)
sim = TransmitBeamplot(**kwargs)
sim.solve()


# rf_data = sim.result['rf_data']
# times = sim.result['times']

