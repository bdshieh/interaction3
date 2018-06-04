## interaction3 / reconstruction / engines_cy.pyx

__all__ = ['time_beamform', 'freq_beamform']

import numpy as np
cimport numpy as np
import cython
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray time_beamform(double[:, :, :] rfdata, double[:, :, :] delays, int window, double fs,
    int[:] channel_mask, double[:] apodization):
    
    cdef int npos = delays.shape[0]
    cdef int ndelay_frames = delays.shape[2]
    cdef int nsamples = rfdata.shape[0]
    cdef int nchannels = rfdata.shape[1]
    cdef int nframes = rfdata.shape[2]
    cdef double[:, :, :] bfdata = np.zeros((npos, window, nframes), dtype=np.float64)
    cdef int winhalf = (window - 1) // 2
    cdef int pos, ch, idx, sample, delay, sd, frame
    cdef int[:, :, :] sample_delays = np.round(np.asarray(delays) * fs).astype(int)

    nsamples = nsamples - 2 * window

    # apply delays in loop over field points and channels
    if ndelay_frames == 1:
        for pos in range(npos):
            for ch in range(nchannels):

                sd = sample_delays[pos, ch, 0]

                if not channel_mask[ch]:
                    continue
                if sd > nsamples + winhalf:
                    continue
                if sd < -(winhalf + 1):
                    continue

                delay = sd + window - winhalf

                for i, sample in enumerate(range(delay, delay + window)):
                    for frame in range(nframes):
                        bfdata[pos, i, frame] += apodization[ch] * rfdata[sample, ch, frame]

    else:
        for frame in range(nframes):
            for pos in range(npos):
                for ch in range(nchannels):

                    sd = sample_delays[pos, ch, frame]

                    if not channel_mask[ch]:
                        continue
                    if sd > nsamples + winhalf:
                        continue
                    if sd < -(winhalf + 1):
                        continue

                    delay = sd + window - winhalf

                    for i, sample in enumerate(range(delay, delay + window)):
                        bfdata[pos, i, frame] += apodization[ch] * rfdata[sample, ch, frame]

    return np.asarray(bfdata)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray freq_beamform(double[:, :, :] rfdata, double[:, :, :] delays, int window, double fs,
    int[:] channel_mask, double[:] apodization):

    pass
    
    
    
    
    
    
    
    
    