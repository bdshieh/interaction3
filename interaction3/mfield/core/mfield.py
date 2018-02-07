# mfield / mfield.py

import numpy as np
import matlab.engine
# import StringIO
#import sys
import os


class MField(object):
    '''
    Implementation of FIELD II using the MATLAB engine for python.
    '''
    def __init__(self, path=None):

        # make sure to set MATLAB engine path to location of m-files
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))

        self._mateng = matlab.engine.start_matlab()
        self._mateng.cd(str(os.path.normpath(path)), nargout=0)

    def numpy_to_mat(self, array, orient='row'):

        if array.ndim == 1:
            if orient.lower() == 'row':
                sz = (1, array.size)
            elif orient.lower() in ('col', 'column'):
                sz = (array.size, 1)
        else:
            sz = None

        ret = matlab.double(initializer=array.tolist(), size=sz)

        return ret

    def close(self):
        self._mateng.quit()

    def mat_to_numpy(self, array):
        return np.array(array).squeeze()

    def field_init(self, suppress=-1):
        self._mateng.field_init(suppress, nargout=0)

    def field_end(self):
        self._mateng.field_end(nargout=0, stdout=StringIO.StringIO())

    def set_field(self, option_name, value):
        self._mateng.set_field(option_name, value, nargout=0)

    def field_info(self):
        self._mateng.field_info(nargout=0)

    def calc_scat(self, Th1, Th2, points, amplitudes):

        points_mat = self.numpy_to_mat(points, orient='row')
        amplitudes_mat = self.numpy_to_mat(amplitudes, orient='col')

        ret = self._mateng.calc_scat(Th1, Th2, points_mat, amplitudes_mat,
            nargout=2)

        scat = self.mat_to_numpy(ret[0])
        t0 = ret[1]

        return scat, t0

    def calc_scat_all(self, Th1, Th2, points, amplitudes, dec_factor):

        points_mat = self.numpy_to_mat(points, orient='row')
        amplitudes_mat = self.numpy_to_mat(amplitudes, orient='col')

        ret = self._mateng.calc_scat_all(Th1, Th2, points_mat, amplitudes_mat,
            dec_factor, nargout=2)

        scat = self.mat_to_numpy(ret[0])
        t0 = ret[1]

        return scat, t0

    def calc_scat_multi(self, Th1, Th2, points, amplitudes):

        points_mat = self.numpy_to_mat(points, orient='row')
        amplitudes_mat = self.numpy_to_mat(amplitudes, orient='col')

        ret = self._mateng.calc_scat_multi(Th1, Th2, points_mat, amplitudes_mat,
            nargout=2)

        scat = self.mat_to_numpy(ret[0])
        t0 = ret[1]

        return scat, t0

    def calc_h(self, Th, points):

        points_mat = self.numpy_to_mat(points, orient='row')

        ret = self._mateng.calc_h(Th, points_mat, nargout=2)

        h = self.mat_to_numpy(ret[0])
        t0 = ret[1]

        return h, t0

    def calc_hp(self, Th, points):

        points_mat = self.numpy_to_mat(points, orient='row')

        ret = self._mateng.calc_hp(Th, points_mat, nargout=2)

        hp = self.mat_to_numpy(ret[0])
        t0 = ret[1]

        return hp, t0

    def calc_hhp(self, Th1, Th2, points):

        points_mat = self.numpy_to_mat(points, orient='row')

        ret = self._mateng.calc_hhp(Th1, Th2, points_mat, nargout=2)

        hhp = self.mat_to_numpy(ret[0])
        t0 = ret[1]

        return hhp, t0

    def xdc_impulse(self, Th, pulse):

        pulse_mat = self.numpy_to_mat(pulse, orient='row')
        self._mateng.xdc_impulse(Th, pulse_mat, nargout=0)

    def xdc_excitation(self, Th, pulse):

        pulse_mat = self.numpy_to_mat(pulse, orient='row')
        self._mateng.xdc_excitation(Th, pulse_mat, nargout=0)

    def xdc_linear_array(self, no_elements, width, height, kerf, no_sub_x,
        no_sub_y, focus):

        focus_mat = self.numpy_to_mat(focus, orient='row')
        ret = self._mateng.xdc_linear_array(no_elements, width, height, kerf,
            no_sub_x, no_sub_y, focus_mat, nargout=1)

        return ret

    def xdc_show(self, Th, info_type='all'):
        self._mateng.xdc_show(Th, info_type, nargout=0)
    
    def xdc_focus(self, Th, times, points):

        times_mat = self.numpy_to_mat(times, orient='col')
        points_mat = self.numpy_to_mat(points, orient='row')

        self._mateng.xdc_focus(Th, times_mat, points_mat, nargout=0)   

    def xdc_focus_times(self, Th, times, delays):

        times_mat = self.numpy_to_mat(times, orient='col')
        delays_mat = self.numpy_to_mat(delays, orient='row')

        self._mateng.xdc_focus_times(Th, times_mat, delays_mat, nargout=0)

    def xdc_free(self, Th):
        self._mateng.xdc_free(Th, nargout=0)

    def xdc_get(self, Th, info_type='rect'):

        ret = self.mat_to_numpy(self._mateng.xdc_get(Th, info_type, nargout=1))
        return ret

    def xdc_rectangles(self, rect, center, focus):

        rect_mat = self.numpy_to_mat(rect, orient='row')
        center_mat = self.numpy_to_mat(center, orient='row')
        focus_mat = self.numpy_to_mat(focus, orient='row')

        ret = self._mateng.xdc_rectangles(rect_mat, center_mat, focus_mat,
            nargout=1)

        return ret

    def xdc_focused_array(self, no_elements, width, height, kerf, rfocus,
        no_sub_x, no_sub_y, focus):

        focus_mat = self.numpy_to_mat(focus, orient='row')

        ret = self._mateng.xdc_focused_array(no_elements, width, height, kerf,
            rfocus, no_sub_x, no_sub_y, focus_mat, nargout=1)

        return ret
    
    def xdc_piston(self, radius, ele_size):
        
        ret = self._mateng.xdc_piston(radius, ele_size)
        
        return ret
        
    def xdc_2d_array(self):
        pass

    def xdc_apodization(self, Th, times, values):

        times_mat = self.numpy_to_mat(times, orient='col')
        values_mat = self.numpy_to_mat(values, orient='row')

        self._mateng.xdc_apodization(Th, times_mat, values_mat, nargout=0)

    def xdc_concave(self):
        pass

    def xdc_convex_array(self):
        pass

    def xdc_convex_focused_array(self):
        pass


if __name__ == '__main__':

    from scipy.signal import gausspulse

    field = MField()

    field.field_init()
    field.set_field('c', 1540)
    field.set_field('fs', 100e6)
    field.set_field('att', 0)
    field.set_field('freq_att', 10e6)
    field.set_field('att_f0', 0)
    field.set_field('use_att', 1)

    fc = 10e6
    fbw = 1.0
    fs = 100e6

    cutoff = gausspulse('cutoff', fc=fc, bw=fbw, tpr=-60, bwr=-3)
    adj_cutoff = np.ceil(cutoff*fs)/fs
    t = np.arange(-adj_cutoff, adj_cutoff + 1/fs, 1/fs)
    _, impulse_response = gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)
    excitation = impulse_response.copy()

    tx = field.xdc_linear_array(64, 0.0002, 0.001, 300e-6, 1, 2,
        np.array([0,0,0.03]))
    field.xdc_impulse(tx, impulse_response)
    field.xdc_excitation(tx, excitation)

    #field.field_info()
    #field.xdc_show(tx)

    scat, t0 = field.calc_scat_multi(tx, tx, np.array([0, 0, 0.03]),
        np.array([1]))

    field.field_end()

    field.close()
