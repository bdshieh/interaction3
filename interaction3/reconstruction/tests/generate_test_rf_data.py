

import numpy as np
from interaction3.mfield.core import mfield
from interaction3.reconstruction.tests import sim_functions as sim



fs = 100e6
c = 1540
use_att = True
att = 0
freq_att = 0
att_f0 = 1e6
excitation_fc = 5e6
excitation_bw = 4e6
field_pos = np.array([0, 0, 0.02])
angles = np.linspace(-40, 40, 17)
nelements = 32
pitch = 300e-6
kerf = 112e-6

field = mfield.MField()

# initialize Field II with parameters
field.field_init()
field.set_field('c', c)
field.set_field('fs', fs)
field.set_field('use_att', use_att)
field.set_field('att', att)
field.set_field('freq_att', freq_att)
field.set_field('att_f0', att_f0)

# create excitation
pulse, _ = sim.gausspulse(excitation_fc, excitation_bw / excitation_fc, fs)

array = field.xdc_linear_array(nelements, pitch - kerf, 1e-2, kerf, 1, 1, np.array([0, 0, 300]))
field.xdc_excitation(array, pulse)
field.xdc_impulse(array, np.ones(1))

planewave_rf, planewave_t0 = field.calc_scat_multi(array, array, field_pos, np.array([1]))
synthetic_rf, synthetic_t0 = field.calc_scat_all(array, array, field_pos, np.array([1]), 1)
synthetic_rf = synthetic_rf.reshape((-1, nelements, nelements), order='F')

angular_rf, angular_t0 = list(), list()
for a in angles:

    x = 3000 * np.sin(np.deg2rad(a))
    y = 0
    z = 3000 * np.cos(np.deg2rad(a))

    field.xdc_focus(array, np.array([0]), np.array([x, y, z]))
    rf_data, t0 = field.calc_scat_multi(array, array, field_pos, np.array([1]))
    angular_rf.append(rf_data)
    angular_t0.append(t0)

angular_rf, angular_t0 = sim.concatenate_with_padding(angular_rf, angular_t0, fs, axis=2)

np.savez('test_rf_data_v2.npz', pulse=pulse, fs=fs, planewave_rf=planewave_rf, planewave_t0=planewave_t0,
         synthetic_rf=synthetic_rf, synthetic_t0=synthetic_t0, angular_rf=angular_rf, angular_t0=angular_t0)


field.field_end()
