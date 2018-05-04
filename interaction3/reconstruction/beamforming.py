# reconstruction / beamformer.py

__all__ = ['beamform']

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist as distance
import attr

try:
    import engines
    _ENGINES_PYX_PRESENT = True
except ImportError:
    _ENGINES_PYX_PRESENT = False


def beamform(rfdata, fieldpos, **kwargs):

    # get data attribtues and beamforming options
    txpos = kwargs.get('txpos', None)
    rxpos = kwargs.get('rxpos')
    nwin = int(kwargs.get('nwin', 51))
    fs = kwargs.get('fs')
    c = kwargs.get('c')
    resample = kwargs.get('resample', False)

    chmask = kwargs.get('chmask', False)
    t0 = kwargs.get('t0', 0.0)
    apod = kwargs.get('apod', False)
    ram_limit = kwargs.get('ram_limit', 2.0)

    # elevate arrays to 2d if needed
    fieldpos, rxpos = np.atleast_2d(fieldpos, rxpos)

    if rfdata.ndim == 3:
        nsample, nchannel, nframe = rfdata.shape
    elif rfdata.ndim == 2:
        nsample, nchannel = rfdata.shape
        rfdata = rfdata[..., None]

    # set defaults
    if chmask is False:
        chmask = np.ones(nchannel)
    if apod is False:
        apod = np.ones(nchannel)
    if resample is False:
        resample = 1

    # resample data
    if resample != 1:
        rfdata = sp.signal.resample(rfdata, nsample * resample)
        nsample = rfdata.shape[0]

    if not nwin % 2:
        nwin += 1

    # pad data
    pad_width = ((nwin, nwin), (0, 0), (0, 0))
    rfdata = np.pad(rfdata, pad_width, mode='constant')

    # select beamforming engine
    if _ENGINES_PYX_PRESENT:
        
        engine = engines._time_beamform_engine
        chmask = chmask.astype(np.int32)
        apod = apod.astype(np.float64)
        
    else:
        engine = _time_beamform_engine

    # calculate transmit delays
    if txpos is None:
        txdelay = np.zeros((fieldpos.shape[0], 1), dtype=np.float64)
    elif txpos in ['plane', 'planetx', 'planewave', 'pw']:
        txdelay = np.abs(fieldpos[:, 2:3]) / c
    else:
        txpos = np.atleast_2d(txpos)
        txdelay = distance(fieldpos, txpos) / c

    # split fieldpos array to reduce memory usage
    npos = fieldpos.shape[0]
    nrx = rxpos.shape[0]
    nsplit = int(np.ceil((npos * nrx * 8 / 1000 / 1000 / 1000) / ram_limit))

    result = list()
    for fp, td in zip(np.array_split(fieldpos, nsplit, axis=0), np.array_split(txdelay, nsplit, axis=0)):

        rd = distance(fp, rxpos) / c
        delays = np.round((td + rd - t0) * fs * resample).astype(np.int32)

        bfdata = engine(rfdata, delays, nwin, chmask, apod)
        result.append(bfdata)

    return np.concatenate(result, axis=0)


@attr.s
class Beamformer:

    rfdata = attr.ib(init=True)
    field_pos = attr.ib(init=True, converter=np.atleast_2d)
    receive_pos = attr.ib(init=True, converter=np.atleast_2d)
    transmit_pos = attr.ib(init=True, converter=np.atleast_2d, default=None)
    planewave = attr.ib(init=True, default=False)
    angles = attr.ib(init=True, default=None)
    window = attr.ib(init=True, default=51)
    sample_frequency = attr.ib(init=True, default=40e6)
    sound_speed = attr.ib(init=True, default=1540)
    apodization = attr.ib(init=True)
    channel_mask = attr.ib(init=True)
    t0 = attr.ib(init=True, default=0)
    resample = attr.ib(init=True, default=1)
    chunksize = attr.ib(init=True, default=100)

    result = attr.ib(init=False, default=attr.Factory(dict))

    @rfdata.converter
    def _rfdata_converter(self, attribute, value):
        if value.ndim == 2:
            return value[..., None]

    @rfdata.validator
    def _rfdata_validator(selfself, attribute, value):
        if value.ndim != 3:
            raise ValueError('rfdata must have dimensions 3')

    @transmit_pos.default
    def _transmit_pos_default(self):
        pass

    @channel_mask.default
    def _channel_mask_default(self):
        nchannels = self.rfdata.shape[1]
        return np.ones(nchannels)

    @apodization.default
    def _apodization_default(self):
        nchannels = self.rfdata.shape[1]
        return np.ones(nchannels)

    @window.converter
    def _window_converter(self, attribute, value):
        if not value % 2:
            value += 1

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
            normals = np.c_[np.cos(rad_angles), np.sin(rad_angles), 0]
            r = field_pos - transmit_pos
            transmit_delays = np.abs(np.dot(r, normals.T)) / sound_speed
            transmit_delays = transmit_delays[:, None, :]

        else:

            if transmit_pos is None: # no transmit delays (assume receive only)
                transmit_delays = np.zeros((npos, nchannels, nframes))

            else:
                transmit_delays = distance(field_pos, transmit_pos) / sound_speed
                transmit_delays = transmit_delays[:, None, :]

        receive_delays = distance(field_pos, receive_pos) / sound_speed
        receive_delays = receive_delays[..., None]

        delays = transmit_delays + receive_delays - t0

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

    pass

    # prms = dict()
    # prms['txpos'] = np.c_[np.linspace(-0.02, 0.02, 676), np.zeros(676), np.zeros(676)]
    # prms['rxpos'] = np.c_[np.linspace(-0.02, 0.02, 676), np.zeros(676), np.zeros(676)]
    # x, y, z = np.mgrid[-0.02:0.02:60j, -0.02:0.02:60j, 0.001:0.041:60j]
    # prms['fieldpos'] = np.c_[x.ravel(), y.ravel(), z.ravel()]
    # prms['rfdata'] = sp.rand(3000, 676)
    # prms['fs'] = 100e6
    # prms['c'] = 1540.
    
    #delays = np.round(sp.rand(1, 676)*100).astype(np.int32)
    #nwin = int(51)
    #chmask = np.ones(676, dtype=np.int32)
    #apod = np.ones(676, dtype=np.float64)
    
    #bfdata = beamform(**prms)

    
