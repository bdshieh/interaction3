 # imaging / engines.pyx

import numpy as np
cimport numpy as np
import scipy as sp

import cython
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray _time_beamform_engine(double[:,:,:] rfdata, int[:,:] delays, int nwin, int[:] chmask, double[:] apod):
    
    cdef int npos = delays.shape[0]
    cdef int nsample = rfdata.shape[0]
    cdef int nchannel = rfdata.shape[1]
    cdef int nframe = rfdata.shape[2]
    cdef double[:,:,:] bfdata = np.zeros((npos, nwin, nframe), dtype=np.float64)
    cdef double[:,:] bfsig = np.zeros((nwin, nframe), dtype=np.float64)
    cdef int nwinhalf = (nwin - 1)/2
    cdef int pos, ch, idx, sample, delay, sd
    
    nsample = nsample - 2*nwin

    # apply delays in loop over field points and channels
    for pos in xrange(npos):
        
        for ch in xrange(nchannel):
            
            sd = delays[pos, ch]
            
            if not chmask[ch]:
                continue
                
            if sd > nsample + nwinhalf:
                continue
            if sd < -(nwinhalf + 1):
                continue

            delay = sd + nwin - nwinhalf
            
            for idx, sample in enumerate(xrange(delay, delay + nwin)):
                
                for frame in xrange(nframe):
                    
                    bfdata[pos, idx, frame] += apod[ch]*rfdata[sample, ch, frame]

    return np.asarray(bfdata)
    
    
    
    
    
    
    
    
    
    