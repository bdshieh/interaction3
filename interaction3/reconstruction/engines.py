
import numpy as np
from scipy import signal

# try:
#     import engines
#     _ENGINES_PYX_PRESENT = True
# except ImportError:
#     _ENGINES_PYX_PRESENT = False


def _time_beamform(rfdata, delays, window, channel_mask, apodization, sample_frequency, resample):

    # get data attribtues and beamforming options
    npos, _, ndelay_frames = delays.shape
    nsamples, nchannels, nframes = rfdata.shape
    nsamples = nsamples - 2 * window
    winhalf = (window - 1) // 2

    if resample > 1:

        # use floor if resampling to keep windowing logic simple
        sample_delays = np.floor(delays * sample_frequency).astype(int)
        resample_delays = np.floor(delays * sample_frequency * resample).astype(int)
        resample_window = winhalf * resample * 2 + 1
        bfdata = np.zeros((npos, resample_window, nframes))
    else:
        sample_delays = np.round(delays * sample_frequency).astype(int)
        bfdata = np.zeros((npos, window, nframes))

    if ndelay_frames == 1:

        for pos in range(npos):

            sd = sample_delays[pos, :, 0]
            valid_delay = (sd <= (nsamples + winhalf)) & (sd >= -(winhalf + 1))

            if not np.any(valid_delay):
                continue

            if resample > 1:
                rsd = resample_delays[pos, :, 0]
                bfsig = np.zeros((resample_window, nframes))
            else:
                bfsig = np.zeros((window, nframes))

            for ch in range(nchannels):

                if not valid_delay[ch]:
                    continue

                if not channel_mask[ch]:
                    continue

                delay = sd[ch] + window - winhalf  # account for zero-padding
                window_rf = rfdata[delay:(delay + window), ch, :]

                if resample > 1:

                    resample_rf = signal.resample(window_rf, window * resample, axis=0)
                    start = rsd[ch] - (resample * sd[ch])
                    rf = resample_rf[start:(start + resample_window), :]
                else:

                    rf = window_rf

                bfsig += apodization[ch] * rf

            bfdata[pos, :, :] = bfsig

    else:

        for frame in range(nframes):

            for pos in range(npos):

                sd = sample_delays[pos, :, frame]
                valid_delay = (sd <= (nsamples + winhalf)) & (sd >= -(winhalf + 1))

                if not np.any(valid_delay):
                    continue

                if resample > 1:
                    rsd = resample_delays[pos, :, frame]
                    bfsig = np.zeros((resample_window, nframes))
                else:
                    bfsig = np.zeros(window)

                for ch in range(nchannels):

                    if not valid_delay[ch]:
                        continue

                    if not channel_mask[ch]:
                        continue

                    delay = sd[ch] + window - winhalf  # account for zero-padding
                    window_rf = rfdata[delay:(delay + window), ch, frame]

                    if resample > 1:

                        resample_rf = signal.resample(window_rf, window * resample, axis=0)
                        start = rsd[ch] - (resample * sd[ch])
                        rf = resample_rf[start:(start + resample_window), frame]
                    else:

                        rf = window_rf

                    bfsig += apodization[ch] * rf

                bfdata[pos, :, frame] = bfsig

    return bfdata
