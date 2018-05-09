# reconstruction / beamformer.py

# __all__ = ['beamform']

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist as distance
import attr

from engines import _time_beamform


def _rfdata_converter(value):
    if value.ndim == 2:
        return value[..., None]


def _window_converter(value):
    if not value % 2:
        value += 1

    return int(value)


@attr.s
class Beamformer(object):

    rfdata = attr.ib(init=True, converter=_rfdata_converter)
    field_pos = attr.ib(init=True, converter=np.atleast_2d)
    receive_pos = attr.ib(init=True, converter=np.atleast_2d)
    transmit_pos = attr.ib(init=True, converter=np.atleast_2d, default=None)
    planewave = attr.ib(init=True, default=False)
    angles = attr.ib(init=True, default=None)
    window = attr.ib(init=True, converter=_window_converter, default=51)
    sample_frequency = attr.ib(init=True, default=40e6)
    sound_speed = attr.ib(init=True, default=1540)
    apodization = attr.ib(init=True)
    channel_mask = attr.ib(init=True)
    t0 = attr.ib(init=True, default=0)
    resample = attr.ib(init=True, default=1)
    chunksize = attr.ib(init=True, default=100)

    result = attr.ib(init=False, default=attr.Factory(dict))

    @rfdata.validator
    def _rfdata_validator(selfself, attribute, value):
        if value.ndim != 3:
            raise ValueError('rfdata must have dimensions 3')

    @channel_mask.default
    def _channel_mask_default(self):
        nchannels = self.rfdata.shape[1]
        return np.ones(nchannels)

    @apodization.default
    def _apodization_default(self):
        nchannels = self.rfdata.shape[1]
        return np.ones(nchannels)

    @transmit_pos.validator
    def _transmit_pos_validator(self, attribute, value):

        if self.planewave:
            if value.shape != (1, 3):
                raise ValueError('For planewave, transmit_pos must be a single position')

    def run(self):

        rfdata = self.rfdata
        field_pos = self.field_pos
        transmit_pos = self.transmit_pos
        receive_pos = self.receive_pos
        window = self.window
        resample = self.resample
        sound_speed = self.sound_speed
        planewave = self.planewave
        angles = self.angles
        sample_frequency = self.sample_frequency
        t0 = self.t0
        channel_mask = self.channel_mask
        apodization = self.apodization

        nsamples, nchannels, nframes = rfdata.shape
        npos, _ = field_pos.shape

        # resample data
        # if resample != 1:
        #     rfdata = sp.signal.resample(rfdata, nsamples * resample)
        #     nsamples = rfdata.shape[0]

        # select beamforming engine
        # if _ENGINES_PYX_PRESENT:
        #     engine = engines._time_beamform_engine
        #     chmask = chmask.astype(np.int32)
        #     apod = apod.astype(np.float64)
        # else:
        #     engine = _time_beamform_engine

        # pad data
        pad_width = ((window, window), (0, 0), (0, 0))
        rfdata = np.pad(rfdata, pad_width, mode='constant')

        # calculate transmit delays
        if planewave:

            if transmit_pos is None:
                transmit_pos = np.zeros((1, 3))

            if angles is None:
                angles = [0,]

            rad_angles = np.deg2rad(angles)
            normals = np.c_[np.cos(rad_angles), np.zeros(len(rad_angles)), np.sin(rad_angles)]
            r = field_pos - transmit_pos
            transmit_delays = np.abs(np.dot(r, normals.T)) / sound_speed
            transmit_delays = transmit_delays[:, None, :]

        else:

            if transmit_pos is None: # no transmit delays (assume receive only)
                transmit_delays = np.zeros((npos, nchannels, nframes))

            else:
                transmit_delays = distance(field_pos, transmit_pos) / sound_speed
                transmit_delays = transmit_delays[:, None, :]

        # calculate receive delays
        receive_delays = distance(field_pos, receive_pos) / sound_speed
        receive_delays = receive_delays[..., None]

        # calculate total delays
        delays = transmit_delays + receive_delays - t0

        bfdata = _time_beamform(rfdata, delays, window, channel_mask, apodization, sample_frequency, resample)

        return bfdata

        # split fieldpos array to reduce memory usage
        # npos = field_pos.shape[0]
        # nrx = receive_pos.shape[0]
        # nsplit = int(np.ceil((npos * nrx * 8 / 1000 / 1000 / 1000) / ram_limit))

        # result = list()
        # for fp, td in zip(np.array_split(field_pos, nsplit, axis=0), np.array_split(txdelay, nsplit, axis=0)):
        #
        #     rd = distance(fp, rxpos) / c
        #     delays = np.round((td + rd - t0) * fs * resample).astype(np.int32)
        #
        #     bfdata = engine(rfdata, delays, nwin, chmask, apod)
        #     result.append(bfdata)

        # return np.concatenate(result, axis=0)



if __name__ == '__main__':

    prms = dict()
    # prms['transmit_pos'] = np.c_[np.linspace(-0.02, 0.02, 16), np.zeros(16), np.zeros(16)]
    prms['transmit_pos'] = [0, 0, 0]
    prms['receive_pos'] = np.c_[np.linspace(-0.02, 0.02, 16), np.zeros(16), np.zeros(16)]
    x, y, z = np.mgrid[-0.02:0.02:11j, 0:1:1j, 0.02:0.04:11j]
    prms['field_pos'] = np.c_[x.ravel(), y.ravel(), z.ravel()]
    prms['rfdata'] = sp.rand(2000, 16)
    prms['planewave'] = True
    # prms['angles'] = np.linspace(-20, 20, 3)
    prms['resample'] = 4

    bf = Beamformer(**prms)
    bfdata = bf.run()
    
