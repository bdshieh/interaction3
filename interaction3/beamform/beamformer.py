# reconstruction / beamformer.py

__all__ = ['Beamformer']

import numpy as np
import scipy as sp
import multiprocessing
from scipy.spatial.distance import cdist as distance
import attr
from tqdm import tqdm

from . engines import *
from . import sim_functions as sim


def _process(job):

    job_id, (rfdata, t0, field_pos, transmit_pos, receive_pos, window, sound_speed, fs, planewave, angles,
             normalize_delays, channel_mask, apodization, return_image) = job

    field_pos = np.array(field_pos)
    nsamples, nchannels, nframes = rfdata.shape
    npos, _ = field_pos.shape

    # calculate transmit delays
    if planewave:

        if transmit_pos is None:
            transmit_pos = np.zeros((1, 3))

        if angles is None:
            angles = [0, ]

        rad_angles = np.deg2rad(angles)
        normals = np.c_[np.sin(rad_angles), np.zeros(len(rad_angles)), np.cos(rad_angles)]
        r = field_pos - transmit_pos
        transmit_delays = np.abs(np.dot(r, normals.T)) / sound_speed
        transmit_delays = transmit_delays[:, None, :]

        # normalize delays by increasing/decreasing all delays so the minimum delay is 0
        if normalize_delays:

            r = receive_pos - transmit_pos
            pw_delay = np.dot(r, normals.T) / sound_speed
            correction = np.min(pw_delay, axis=0)[None, None, :]
            transmit_delays += correction

    else:

        if transmit_pos is None:  # no transmit delays (assume receive only)
            transmit_delays = np.zeros((npos, nchannels, nframes))

        else:  # full synthetic transmit
            transmit_delays = distance(field_pos, transmit_pos) / sound_speed
            transmit_delays = transmit_delays[:, None, :]

    # calculate receive delays
    receive_delays = distance(field_pos, receive_pos) / sound_speed
    receive_delays = receive_delays[..., None]

    # calculate total delays
    delays = transmit_delays + receive_delays - t0

    # run beamformer
    bfdata = time_beamform(rfdata, delays, window, fs, channel_mask, apodization)

    if return_image:

        imgdata = sim.envelope(np.sum(bfdata, axis=-1), axis=1)[:, window // 2]
        return job_id, imgdata

    else:
        return job_id, bfdata


def _rfdata_converter(value):
    if value.ndim == 2:
        return value[..., None]
    else:
        return value


def _window_converter(value):
    if not value % 2:
        value += 1
    return int(value)


@attr.s
class Beamformer(object):
    """
    A generalized delay-and-sum beamformer.

    Parameters
    ----------
    rfdata : array_like, shape (M, N, P)
        A 2-D or 3-D RF data matrix with M samples, N channels, and P frames.
    field_pos : array_like, shape (..., 3)
        A 2-D array of field positions to beamform to.
    receive_pos : array_like, shape (N, 3)
        A 2-D array of receive positions for each channel
    transmit_pos : array_like, shape (..., 3)
        A 2-D array of transmit positions. If planewave is True, this is interpreted as the origin location of the
        planewave. Otherwise, each transmit position corresponds to a frame, where the sequence is repeated if the
        number of transmit positions is less than the number of frames.
    planewave: bool, optional
        Assume planewave in transmit beamforming, default is False.
    angles: array_like or None, optional
        A 1-D array of angles corresponding to a frame for angular compounding. Default is None, which ignores
        compounding.
    window: int, optional
        Number of samples in window of returned beamformed data. Default is 51.
    sample_frequency: float, optional
        Sampling rate in samples per second. Default is 40e6.
    sound_speed: float, optional
        Speed of sound in meters per second. Default is 1540.
    t0: float, optional
        Time corresponding to the first sample in seconds. Default is 0.
    resample: int, optional
        Integer factor to resample RF data. Default is 1.
    apodization: array_like, optional
        A 1-D array of apodization values for each channel. Default is an array of ones.
    channel_mask: array_like, optional
        A 1-D array of boolean values determining whether a channel is used (True) or ignored (False). Default is an
        array of True.
    normalize_delays: bool, optional
        Normalize delays for each field position by shifting so that the minimum delay is zero. Default is True.
    ram_limit: float, optional
        Limits RAM usage (approximately) to this value in GB. Default is 1.
    threads: int, optional
        Number of threads. Default is number of logical cores available.
    return_image: bool, optional
        Return the image only, i.e. the envelope-detected brightness, instead of beamformed data. Greatly reduces
        RAM usage. Default is True.



    Methods
    -------
    run
    """
    rfdata = attr.ib(converter=_rfdata_converter)
    field_pos = attr.ib(converter=np.atleast_2d)
    receive_pos = attr.ib(converter=np.atleast_2d)

    transmit_pos = attr.ib(converter=np.atleast_2d, default=None)
    planewave = attr.ib(default=False)
    angles = attr.ib(default=None)
    window = attr.ib(converter=_window_converter, default=51)
    sample_frequency = attr.ib(default=40e6)
    sound_speed = attr.ib(default=1540)
    t0 = attr.ib(default=0)
    resample = attr.ib(default=1)
    apodization = attr.ib()
    channel_mask = attr.ib()
    normalize_delays = attr.ib(default=True)
    ram_limit = attr.ib(default=0.5)
    threads = attr.ib(default=multiprocessing.cpu_count())
    return_image = attr.ib(default=True)

    # result = attr.ib(init=False, default=attr.Factory(dict))

    @rfdata.validator
    def _rfdata_validator(selfself, attribute, value):
        if value.ndim != 3:
            raise ValueError('rfdata must have dimensions 3')

    @channel_mask.default
    def _channel_mask_default(self):
        nchannels = self.rfdata.shape[1]
        return np.ones(nchannels, dtype=int)

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
        """
        Runs the beamformer.

        Returns
        -------
        Result : array_like, shape (M, ..., P)
            Result of the beamforming operation where M is the number of field positions and P is the number of frames.
            If return_image is True, returns image data with shape (M, 1, P). Otherwise, returns beamformed data with
            (M, N, P) where N is the number of samples in the window.
        """
        rfdata = self.rfdata
        field_pos = self.field_pos
        transmit_pos = self.transmit_pos
        receive_pos = self.receive_pos
        window = self.window
        resample = self.resample
        sound_speed = self.sound_speed
        planewave = self.planewave
        angles = self.angles
        fs = self.sample_frequency
        t0 = self.t0
        channel_mask = self.channel_mask
        apodization = self.apodization
        normalize_delays = self.normalize_delays
        ram_limit = self.ram_limit
        threads = self.threads
        return_image = self.return_image

        nsamples, nchannels, nframes = rfdata.shape
        npos, _ = field_pos.shape

        # upsample data
        if resample > 1:
            rfdata = sp.signal.resample(rfdata, nsamples * resample, axis=0)
            fs = fs * resample
            window = (window - 1) // 2 * resample * 2 + 1

        # pad data
        pad_width = ((window, window), (0, 0), (0, 0))
        rfdata = np.pad(rfdata, pad_width, mode='constant')

        # split field_pos into chunks to avoid excessive RAM usage when pre-calculating delays
        # njobs = int(np.ceil(npos * nchannels * nframes * 8 / 1024 / 1024 / 1024 / ram_limit))
        njobs = int(np.ceil(npos * window * nframes * 8 / 1024 / 1024 / 1024 / ram_limit))
        chunksize = int(np.ceil(npos / njobs))

        # create jobs
        jobs = sim.create_jobs(rfdata, t0, (field_pos, chunksize), transmit_pos, receive_pos, window, sound_speed, fs,
                               planewave, angles, normalize_delays, channel_mask, apodization, return_image, mode='zip')

        # create pool and run jobs
        pool = multiprocessing.Pool(threads)
        result = [None] * njobs
        results = pool.imap_unordered(_process, jobs)

        for job_id, r in tqdm(results, desc='Beamforming', total=njobs):
            result[job_id] = r

        return np.concatenate(result, axis=0)


if __name__ == '__main__':
    pass

    # prms = dict()
    # # prms['transmit_pos'] = np.c_[np.linspace(-0.02, 0.02, 16), np.zeros(16), np.zeros(16)]
    # prms['transmit_pos'] = [0, 0, 0]
    # prms['receive_pos'] = np.c_[np.linspace(-0.02, 0.02, 16), np.zeros(16), np.zeros(16)]
    # x, y, z = np.mgrid[-0.02:0.02:11j, 0:1:1j, 0.02:0.04:11j]
    # prms['field_pos'] = np.c_[x.ravel(), y.ravel(), z.ravel()]
    # prms['rfdata'] = sp.rand(2000, 16)
    # prms['planewave'] = True
    # # prms['angles'] = np.linspace(-20, 20, 3)
    # prms['resample'] = 4
    #
    # bf = Beamformer(**prms)
    # bfdata = bf.run()
    
