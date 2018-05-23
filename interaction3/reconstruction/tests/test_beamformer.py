

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from interaction3.reconstruction import sim_functions as sim
from interaction3.reconstruction import Beamformer

filename = 'test_rf_data.npz'
nelements = 32
pitch = 300e-6
angles = np.linspace(-5, 5, 21)

xx, yy, zz = np.mgrid[-0.02:0.02:81j, 0:1:1j, 0.001:0.041:81j]
field_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

xv, yv, zv = np.linspace(0, (nelements - 1) * pitch, nelements) - (nelements - 1) * pitch / 2, 0, 0
xx, yy, zz = np.meshgrid(xv, yv, zv)
array_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]


def test_planewave_beamformer():

    with np.load(filename) as file:
        rfdata = file['planewave_rf']
        t0 = file['planewave_t0']

    kwargs = dict()
    kwargs['sample_frequency'] = 100e6
    kwargs['t0'] = t0
    kwargs['window'] = 101
    kwargs['transmit_pos'] = [0, 0, 0]
    kwargs['receive_pos'] = array_pos
    kwargs['field_pos'] = field_pos
    kwargs['rfdata'] = rfdata
    kwargs['planewave'] = True
    kwargs['resample'] = 8

    bf = Beamformer(**kwargs)
    bfdata = bf.run()
    envdata = sim.envelope(bfdata, axis=1)
    imgdata = np.max(envdata, axis=1).reshape((81, 81))

    plt.figure()
    plt.imshow(imgdata)

    return bfdata, imgdata


def test_synthetic_beamformer():

    with np.load(filename) as file:
        rfdata = file['synthetic_rf']
        t0 = file['synthetic_t0']

    kwargs = dict()
    kwargs['sample_frequency'] = 100e6
    kwargs['t0'] = t0
    kwargs['window'] = 101
    kwargs['transmit_pos'] = array_pos
    kwargs['receive_pos'] = array_pos
    kwargs['field_pos'] = field_pos
    kwargs['rfdata'] = rfdata
    kwargs['planewave'] = False
    kwargs['resample'] = 8

    bf = Beamformer(**kwargs)
    bfdata = bf.run()
    envdata = sim.envelope(np.sum(bfdata, axis=-1), axis=1)
    # envdata = sim.envelope(bfdata, axis=1)
    imgdata = np.max(envdata, axis=1).reshape((81, 81, -1))

    plt.figure()
    plt.imshow(imgdata[..., 0])

    return bfdata, imgdata


def test_angular_beamformer():

    with np.load(filename) as file:
        rfdata = file['angular_rf']
        t0 = file['angular_t0']

    kwargs = dict()
    kwargs['sample_frequency'] = 100e6
    kwargs['t0'] = t0
    kwargs['window'] = 101
    kwargs['transmit_pos'] = [0, 0, 0]
    kwargs['receive_pos'] = array_pos
    kwargs['field_pos'] = field_pos
    kwargs['rfdata'] = rfdata
    kwargs['planewave'] = True
    kwargs['angles'] = angles
    kwargs['resample'] = 1
    kwargs['normalize_delays'] = True

    bf = Beamformer(**kwargs)
    bfdata = bf.run()
    envdata = sim.envelope(np.sum(bfdata, axis=-1), axis=1)
    # envdata = sim.envelope(bfdata, axis=1)
    imgdata = np.max(envdata, axis=1).reshape((81, 81, -1))

    plt.figure()
    plt.imshow(imgdata[..., 0])
    plt.show()

    return bfdata, imgdata


if __name__ == '__main__':

    # pass
    test_planewave_beamformer()
    test_synthetic_beamformer()
    # test_angular_beamformer()
    plt.show()