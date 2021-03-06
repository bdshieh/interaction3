## interaction3 / mfield / solvers / transmit_beamplot.py

import numpy as np
import attr

from interaction3 import abstract, util
from .. core import mfield


@attr.s
class TransmitBeamplot2(object):

    # INSTANCE VARIABLES, FIELD II PARAMETERS
    c = attr.ib()
    fs = attr.ib()
    use_att = attr.ib()
    att = attr.ib()
    freq_att = attr.ib()
    att_f0 = attr.ib()
    rectangles_info = attr.ib(repr=False)

    # INSTANCE VARIABLES, OTHER PARAMETERS
    excitation_fc = attr.ib()
    excitation_bw = attr.ib()
    field_pos = attr.ib(default=None, repr=False)

    # INSTANCE VARIABLES, NO INIT
    _field = attr.ib(init=False, repr=False)
    result = attr.ib(init=False, default=attr.Factory(dict), repr=False)

    @_field.default
    def _field_default(self):

        rect_info = self.rectangles_info

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

        # read rect_info and create list of transmit
        tx_apertures = []

        for info in rect_info:

            tx_info = info['tx_info']

            if tx_info is not None:

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

                # add aperture to list
                tx_apertures.append(tx)

        self._tx_apertures = tx_apertures

        return field

    def solve(self, field_pos=None):

        field = self._field
        tx_apertures = self._tx_apertures
        fs = self.fs
        result = self.result

        if field_pos is None:
            field_pos = self.field_pos
        field_pos = np.atleast_2d(field_pos)

        # iterate over field positions
        tx_p, tx_t0 = [], []

        for tx in tx_apertures:

            _p, _t0 = field.calc_hp(tx, field_pos)
            tx_p.append(_p.T)
            tx_t0.append(_t0)

        p, t0 = util.sum_with_padding(tx_p, tx_t0, fs)
        brightness = np.max(util.envelope(p, N=util.nextpow2(p.shape[1]), axis=1), axis=1)

        # result['p'] = tx_p
        # result['t0'] = tx_t0
        result['brightness'] = brightness

    @staticmethod
    def get_objects_from_spec(*files):
        spec = []

        for file in files:
            obj = abstract.load(file)
            if isinstance(obj, list):
                spec += obj
            else:
                spec.append(obj)

        if len(spec) < 2:
            raise Exception

        arrays = []
        for obj in spec:
            if isinstance(obj, abstract.Array):
                arrays.append(obj)
            elif isinstance(obj, abstract.MfieldSimulation):
                simulation = obj

        return [simulation,] + arrays

    @staticmethod
    def connector(simulation, *arrays):

        # set simulation defaults
        use_att = simulation.get('use_attenuation', False)
        att = simulation.get('attenuation', 0)
        freq_att = simulation.get('frequency_attenuation', 0)
        att_f0 = simulation.get('attenuation_center_frequency', 1e6)
        field_pos = simulation.get('field_positions', None)
        tx_focus = simulation.get('transmit_focus', None)
        c = simulation.get('sound_speed', 1540)
        delay_quant = simulation.get('delay_quantization', None)

        rectangles_info = []
        for array in arrays:

            if tx_focus is not None:  # apply transmit focus
                abstract.focus_array(array, tx_focus, c, delay_quant, kind='tx')
            tx_info = _construct_rectangles_info(array, kind='tx')
            rectangles_info.append(dict(tx_info=tx_info))

        output = {}
        output['rectangles_info'] = rectangles_info
        output['c'] = simulation['sound_speed']
        output['fs'] = simulation['sampling_frequency']
        output['use_att'] = use_att
        output['att'] = att
        output['freq_att'] = freq_att
        output['att_f0'] = att_f0
        output['field_pos'] = field_pos
        output['excitation_fc'] = simulation['excitation_center_frequecy']
        output['excitation_bw'] = simulation['excitation_bandwidth']

        meta = {}

        return output, meta


def _construct_rectangles_info(array, kind='tx'):

    # create lists to store info about each mathematical element in the array
    channel_centers = []
    channel_delays = []
    channel_apodizations = []
    rectangles = []
    ele_delays = []

    if kind.lower() in ['tx', 'transmit']:
        channels = [ch for ch in array['channels'] if ch['kind'].lower() in ['tx', 'transmit', 'both', 'txrx']]
    elif kind.lower() in ['rx', 'receive']:
        channels = [ch for ch in array['channels'] if ch['kind'].lower() in ['rx', 'receive', 'both', 'txrx']]

    for ch_no, ch in enumerate(channels):

        # pull channel properties
        ch_center = ch['position']  # required property
        ch_delay = ch.get('delay', 0)  # optional property
        ch_apod = ch.get('apodization', 1)  # optional property

        channel_centers.append(ch_center)
        channel_delays.append(ch_delay)
        channel_apodizations.append(ch_apod)

        ele_delays_row = []

        for elem in ch['elements']:

            # pull element properties
            elem_delay = elem.get('delay', 0)  # optional property
            elem_apod = elem.get('apodization', 1)  # optional property

            for mem in elem['membranes']:

                # pull membrane properties
                mem_center = mem['position']  # required property
                length_x = mem['length_x']  # required property
                length_y = mem['length_y']  # required property
                ndiv_x = mem.get('ndiv_x', 2)  # optional property
                ndiv_y = mem.get('ndiv_y', 2)  # optional property
                mem_delay = mem.get('delay', 0)  # optional property
                mem_apod = mem.get('apodization', 1)  # optional property
                rotations = mem.get('rotations', None)  # optional property

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

                    rectangles_row.append(ch_no + 1)
                    rectangles_row += list(vert00)
                    rectangles_row += list(vert01)
                    rectangles_row += list(vert11)
                    rectangles_row += list(vert10)
                    rectangles_row.append(elem_apod * mem_apod)
                    rectangles_row.append(ele_length_x)
                    rectangles_row.append(ele_length_y)
                    rectangles_row += list(center)

                    ele_delays_row.append(elem_delay + mem_delay)

                    rectangles.append(rectangles_row)

        ele_delays.append(ele_delays_row)

    rectangles = np.array(rectangles)
    channel_centers = np.array(channel_centers)
    channel_delays = np.array(channel_delays)
    channel_apodizations = np.array(channel_apodizations)
    ele_delays = np.array(ele_delays)

    output = {}
    output['rectangles'] = rectangles
    output['centers'] = channel_centers
    output['delays'] = channel_delays
    output['apodizations'] = channel_apodizations
    output['ele_delays'] = ele_delays

    return output