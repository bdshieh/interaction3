
import numpy as np

try:
    import engines
    _ENGINES_PYX_PRESENT = True
except ImportError:
    _ENGINES_PYX_PRESENT = False


def _time_beamform(rfdata, delays, nwin, channel_mask, apodization):

    # get data attribtues and beamforming options
    npos = delays.shape[0]
    nsamples, nchannels, nframes = rfdata.shape
    nsamples = nsamples - 2 * nwin
    nwinhalf = (nwin - 1) // 2

    bfdata = np.zeros((npos, nwin, nframes))

    # apply delays in loop over field points and channels
    for pos in range(npos):

        sd = delays[pos, :]
        valid_delay = (sd <= (nsamples + nwinhalf)) & (sd >= -(nwinhalf + 1))

        if not np.any(valid_delay):
            continue

        bfsig = np.zeros((nwin, nframes))

        for ch in range(nchannels):

            if not valid_delay[ch]:
                continue

            if not channel_mask[ch]:
                continue

            delay = sd[ch] + nwin - nwinhalf

            bfsig += apodization[ch] * rfdata[delay:(delay + nwin), ch, :]

        bfdata[pos, :, :] = bfsig

    return bfdata
