## mfield / simulations / transmit_receive_beamplot.py

import numpy as np
import attr
from scipy.interpolate import interp2d

from interaction3 import abstract, util
from .. core import mfield


@attr.s
class TransmitBeamplot(object):

    # INSTANCE VARIABLES, FIELD II PARAMETERS
    c = attr.ib()
    fs = attr.ib()
    use_att = attr.ib()
    att = attr.ib()
    freq_att = attr.ib()
    att_f0 = attr.ib()
    rectangles_info = attr.ib(repr=False)

    # INSTANCE VARIABLES, OTHER PARAMETERS
    use_element_factor = attr.ib()
    element_factor_file = attr.ib(repr=False)
    excitation_fc = attr.ib()
    excitation_bw = attr.ib()
    field_pos = attr.ib(default=None, repr=False)

    # INSTANCE VARIABLES, NO INIT
    _field = attr.ib(init=False, repr=False)
    _interpolator = attr.ib(init=False, repr=False)
    result = attr.ib(init=False, default=attr.Factory(dict), repr=False)
    Config = attr.ib(init=False, repr=False)

    @Config.default
    def _Config_default(self):

        _Config = {}
        _Config['use_attenuation'] = False
        _Config['attenuation'] = 0
        _Config['frequency_attenuation'] = 0
        _Config['attenuation_center_frequency'] = 1e6
        _Config['use_element_factor'] = False
        _Config['element_factor_file'] = None
        _Config['field_positions'] = None
        _Config['transmit_focus'] = None
        _Config['sound_speed'] = 1500.
        _Config['delay_quantization'] = None
        return abstract.register_type('Config', _Config)

    @_field.default
    def _field_default(self):

        field = mfield.MField()

        # initialize Field II with parameters
        field.field_init()
        field.set_field('c', self.c)
        field.set_field('fs', self.fs)
        field.set_field('use_att', self.use_att)
        field.set_field('att', self.att)
        field.set_field('freq_att', self.freq_att)
        field.set_field('att_f0', self.att_f0)

        # create excitation
        pulse, _ = util.gausspulse(self.excitation_fc, self.excitation_bw / self.excitation_fc, self.fs)
        self._pulse = pulse

        return field

    @_interpolator.default
    def _interpolator_default(self):
        if self.use_element_factor:
            with np.load(self.element_factor_file) as root:
                alpha = np.linspace(0, 90, 91)
                beta = np.linspace(0, 90, 91)
                correction_db = root['correction_db']

                fcubic = interp2d(alpha, beta, 10 ** (correction_db / 20.), kind='cubic')

            def interpolator(a, b):
                n = len(a)
                z = np.zeros(n)

                for i, (ai, bi) in enumerate(zip(a, b)):
                    z[i] = fcubic(ai, bi)
                return z

            return interpolator
        else:
            return

    def solve(self, field_pos=None):

        field = self._field
        rect_info = self.rectangles_info
        pulse = self._pulse
        use_element_factor = self.use_element_factor
        interpolator = self._interpolator
        fs = self.fs
        result = self.result

        if field_pos is None:
            field_pos = self.field_pos
        field_pos = np.atleast_2d(field_pos)

        array_rf = []
        array_t0s = []

        for info in rect_info:

            tx_info = info['tx_info']
            tx_rectangles = tx_info['rectangles']
            tx_centers = tx_info['centers']
            tx_delays = tx_info['delays']
            tx_apod = tx_info['apodizations']
            tx_ele_delays = tx_info['ele_delays']

            # create transmit aperture and set aperture parameters
            tx = field.xdc_rectangles(tx_rectangles, tx_centers, np.array([[0, 0, 300]]))
            field.xdc_impulse(tx, pulse)
            field.xdc_excitation(tx, np.array([1]))
            field.xdc_focus_times(tx, np.zeros((1, 1)), tx_delays)
            field.xdc_apodization(tx, np.zeros((1, 1)), tx_apod)

            # set mathematical element delays
            field.ele_delay(tx, np.arange(len(tx_ele_delays)) + 1, tx_ele_delays)

            pos_rf = []
            pos_t0s = []

            for i, pos in enumerate(field_pos):
                if use_element_factor:
                    # calculate element factor corrections
                    r_tx = np.atleast_2d(pos) - np.atleast_2d(tx_centers)
                    r, a, b = util.cart2sec(r_tx).T
                    tx_correction = interpolator(np.rad2deg(np.abs(a)), np.rad2deg(np.abs(b)))
                    # apply correction as apodization
                    field.xdc_apodization(tx, np.zeros((1, 1)), tx_apod * tx_correction)

                _rf, _t0 = field.calc_hp(tx, pos)

                pos_rf.append(_rf)
                pos_t0s.append(_t0)

            _array_rf, _array_t0 = util.concatenate_with_padding(pos_rf, pos_t0s, fs, axis=0)

            array_rf.append(_array_rf)
            array_t0s.append(_array_t0)

            field.xdc_free(tx)

        rf_data, t0 = util.sum_with_padding(array_rf, array_t0s, fs)
        times = t0 + np.arange(rf_data.shape[1]) / fs

        result['rf_data'] = rf_data
        result['times'] = times

    @classmethod
    def from_abstract(cls, cfg, *arrays):

        tx_focus = cfg.transmit_focus
        c = cfg.sound_speed
        delay_quant = cfg.delay_quantization

        rectangles_info = []
        for array in arrays:
            if tx_focus is not None:  # apply transmit focus
                abstract.focus_array(array, tx_focus, c, delay_quant, kind='tx')
            tx_info = _construct_rectangles_info(array, kind='tx')
            rectangles_info.append(dict(tx_info=tx_info))

        args = {}
        args['rectangles_info'] = rectangles_info
        args['c'] = cfg.sound_speed
        args['fs'] = cfg.sampling_frequency
        args['use_att'] = cfg.use_att
        args['att'] = cfg.att
        args['freq_att'] = cfg.freq_att
        args['att_f0'] = cfg.att_f0
        args['use_element_factor'] = cfg.use_element_factor
        args['element_factor_file'] = cfg.element_factor_file
        args['field_pos'] = cfg.field_pos
        args['excitation_fc'] = cfg.excitation_center_frequecy
        args['excitation_bw'] = cfg.excitation_bandwidth

        obj = cls(**args)
        obj.metadata = {}
        return obj


def _construct_rectangles_info(array, kind='tx'):

    # create lists to store info about each mathematical element in the array
    elements_centers = []
    elements_delays = []
    elements_apodizations = []
    rectangles = []
    ele_delays = []

    elements = abstract.get_elements_from_array(array, kind=kind)

    for elem_no, elem in enumerate(elements):

        # pull elements properties
        elem_center = elem.position  
        elem_delay = elem.delay  
        elem_apod = elem.apodization  
        elements_centers.append(elem_center)
        elements_delays.append(elem_delay)
        elements_apodizations.append(elem_apod)

        ele_delays_row = []

        for mem in elements.membranes:

            # pull membrane properties
            mem_center = mem.position  
            length_x = mem.length_x  
            length_y = mem.length_y  
            ndiv_x = mem.ndiv_x  
            ndiv_y = mem.ndiv_y  
            mem_delay = mem.delay  
            mem_apod = mem.apodization  
            rotations = mem.rotations  

            ele_length_x = length_x / ndiv_x
            ele_length_y = length_y / ndiv_y

            # use meshgrid to determine centers of mathematical elements
            xv = np.linspace(-length_x / 2 + ele_length_x / 2, length_x / 2 - ele_length_x / 2, ndiv_x)
            yv = np.linspace(-length_y / 2 + ele_length_y / 2, length_y / 2 - ele_length_y / 2, ndiv_y)
            zv = 0
            x, y, z = np.meshgrid(xv, yv, zv, indexing='xy')
            ele_centers = np.c_[x.ravel(), y.ravel(), z.ravel()]

            # apply rotations
            if rotations is not None:
                for vec, angle in rotations:
                    ele_centers = util.rotate_nodes(ele_centers, vec, angle)

            ele_centers += mem_center

            # loop over mathematical elements
            for center in ele_centers:

                # determine vertices in clock-wise order starting from the lower left
                vert00 = np.array([-ele_length_x / 2, -ele_length_y / 2, 0])
                vert01 = np.array([-ele_length_x / 2, ele_length_y / 2, 0])
                vert11 = np.array([ele_length_x / 2, ele_length_y / 2, 0])
                vert10 = np.array([ele_length_x / 2, -ele_length_y / 2, 0])

                # apply rotations
                if rotations is not None:
                    for vec, angle in rotations:
                        vert00 = util.rotate_nodes(vert00, vec, angle)
                        vert01 = util.rotate_nodes(vert01, vec, angle)
                        vert11 = util.rotate_nodes(vert11, vec, angle)
                        vert10 = util.rotate_nodes(vert10, vec, angle)

                vert00 += center
                vert01 += center
                vert11 += center
                vert10 += center

                # create rectangles and ele_delays, order matters!
                rectangles_row = []
                rectangles_row.append(elem_no + 1)
                rectangles_row += list(vert00)
                rectangles_row += list(vert01)
                rectangles_row += list(vert11)
                rectangles_row += list(vert10)
                rectangles_row.append(mem_apod)
                rectangles_row.append(ele_length_x)
                rectangles_row.append(ele_length_y)
                rectangles_row += list(center)

                ele_delays_row.append(mem_delay)
                rectangles.append(rectangles_row)

        ele_delays.append(ele_delays_row)

    rectangles = np.array(rectangles)
    elements_centers = np.array(elements_centers)
    elements_delays = np.array(elements_delays)
    elements_apodizations = np.array(elements_apodizations)
    ele_delays = np.array(ele_delays)

    output = {}
    output['rectangles'] = rectangles
    output['centers'] = elements_centers
    output['delays'] = elements_delays
    output['apodizations'] = elements_apodizations
    output['ele_delays'] = ele_delays

    return output

