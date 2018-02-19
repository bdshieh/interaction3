## mfield / simulations / beamplot_transmit_simulation.py

import numpy as np
import attr
from scipy.interpolate import interp2d

from interaction3 import abstract
from .. core import mfield
from . import sim_functions as sim


@attr.s
class TransmitBeamplot(object):

    # INSTANCE VARIABLES, FIELD II PARAMETERS
    c = attr.ib()
    fs = attr.ib()
    use_att = attr.ib()
    att = attr.ib()
    freq_att = attr.ib()
    att_f0 = attr.ib()
    rectangles = attr.ib(repr=False)
    centers = attr.ib(repr=False)
    delays = attr.ib(repr=False)
    apodizations = attr.ib(repr=False)
    ele_delays = attr.ib(repr=False)

    # INSTANCE VARIABLES, OTHER PARAMETERS
    use_element_factor = attr.ib()
    element_factor_file = attr.ib(repr=False)
    excitation_fc = attr.ib()
    excitation_bw = attr.ib()
    field_pos = attr.ib(default=None, repr=False)

    # INSTANCE VARIABLES, PRIVATE
    _field = attr.ib(init=False, repr=False)
    _interpolator = attr.ib(init=False, repr=False)
    _result = attr.ib(init=False, default=attr.Factory(dict), repr=False)

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
        pulse, _ = sim.gausspulse(self.excitation_fc, self.excitation_bw / self.excitation_fc, self.fs)

        # create transmit aperture and set aperture parameters
        tx = field.xdc_rectangles(self.rectangles, self.centers, np.array([[0, 0, 300]]))
        field.xdc_impulse(tx, pulse)
        field.xdc_excitation(tx, np.array([1]))
        field.xdc_focus_times(tx, np.zeros((1, 1)), self.delays)
        field.xdc_apodization(tx, np.zeros((1, 1)), self.apodizations)

        # set mathematical element delays
        field.ele_delays(tx, np.arange(len(self.ele_delays)) + 1, self.ele_delays)

        self._tx = tx

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
        tx = self._tx
        apododizations = self.apodizations
        centers = self.centers
        use_element_factor = self.use_element_factor
        interpolator = self._interpolator
        fs = self.fs
        result = self.result

        if field_pos is None:
            field_pos = self.field_pos

        rf_data = list()
        times = list()
        # t0s = list()


        for i, pos in enumerate(field_pos):

            if use_element_factor:

                # calculate element factor corrections
                r_tx = np.atleast_2d(pos) - np.atleast_2d(centers)

                r, a, b = sim.cart2sec(r_tx).T
                tx_correction = interpolator(np.rad2deg(np.abs(a)), np.rad2deg(np.abs(b)))

                # apply correction as apodization
                field.xdc_apodization(tx, np.zeros((1, 1)), apododizations * tx_correction)

            rf, t0 = field.calc_hp(tx, pos)

            rf_data.append(rf)
            times.append(t0 + np.arange(len(rf)) / fs)
            # t0s.append(t0)

        result['rf_data'] = rf_data
        result['times'] = times
        # result['t0s'] = t0s


def get_objects_from_spec(*files):

    spec = list()

    for file in files:
        obj = abstract.load(file)
        if isinstance(obj, list):
            spec += obj
        else:
            spec.append(obj)

    if len(spec) != 2:
        raise Exception

    for obj in spec:
        if isinstance(obj, abstract.Array):
            array = obj
        elif isinstance(obj, abstract.BemArrayTransmitSimulation):
            simulation = obj

    return simulation, array


def connector(simulation, array):

    # set simulation defaults
    use_att = simulation.get('use_attenuation', False)
    att = simulation.get('attenuation', 0)
    freq_att = simulation.get('frequency_attenuation', 0)
    att_f0 = simulation.get('attenuation_center_frequency', 1e6)
    use_element_factor = simulation.get('use_element_factor', False)
    field_pos = simulation.get('field_positions', None)

    # create lists to store info about each mathematical element in the array
    channel_centers = list()
    channel_delays = list()
    channel_apodizations = list()
    rectangles = list()
    ele_delays = list()

    for ch_no, ch in enumerate(array['channels']):

        rectangles_row = list()
        ele_delays_row = list()

        # pull channel properties
        ch_center = ch['center'] # required property
        ch_delay = ch.get('delay', 0) # optional property
        ch_apod = ch.get('apodization', 1) # optional property

        channel_centers.append(ch_center)
        channel_delays.append(ch_delay)
        channel_apodizations.append(ch_apod)

        for elem in ch['elements']:

            # pull element properties
            elem_delay = elem.get('delay', 0) # optional property
            elem_apod = elem.get('apodization', 1) # optional property

            for mem in elem['membranes']:

                # pull membrane properties
                mem_center = mem['center'] # required property
                length_x = mem['length_x'] # required property
                length_y = mem['length_y'] # required property
                ndiv_x = mem.get('ndiv_x', 2) # optional property
                ndiv_y = mem.get('ndiv_y', 2) # optional property
                mem_delay = mem.get('delay', 0) # optional property
                mem_apod = mem.get('apodization', 1) # optional property
                rotations = mem.get('rotations', None) # optional property

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
                        ele_centers = sim.rotate_nodes(ele_centers, vec, angle)

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
                            vert00 = sim.rotate_nodes(vert00, vec, angle)
                            vert01 = sim.rotate_nodes(vert01, vec, angle)
                            vert11 = sim.rotate_nodes(vert11, vec, angle)
                            vert10 = sim.rotate_nodes(vert10, vec, angle)

                    vert00 += center
                    vert01 += center
                    vert11 += center
                    vert10 += center

                    # update rectangles, order matters!
                    rectangles_row.append(ch_no + 1)
                    rectangles_row += vert00
                    rectangles_row += vert01
                    rectangles_row += vert11
                    rectangles_row += vert10
                    rectangles_row.append(elem_apod * mem_apod)
                    rectangles_row.append(ele_length_x)
                    rectangles_row.append(ele_length_y)
                    rectangles_row += center

                    ele_delays_row.append(elem_delay + mem_delay)

        rectangles.append(rectangles_row)
        ele_delays.append(ele_delays_row)

    rectangles = np.array(rectangles)
    channel_centers = np.array(channel_centers)
    channel_delays = np.array(channel_delays)
    channel_apodizations = np.array(channel_apodizations)
    ele_delays = np.array(ele_delays)

    output = dict()
    output['rectangles'] = rectangles
    output['centers'] = channel_centers
    output['delays'] = channel_delays
    output['apodizations'] = channel_apodizations
    output['ele_delays'] = ele_delays
    output['c'] = simulation['sound_speed']
    output['use_att'] = use_att
    output['att'] = att
    output['freq_att'] = freq_att
    output['att_f0'] = att_f0
    output['use_element_factor'] = use_element_factor
    output['field_pos'] = field_pos
    output['excitation_fc'] = simulation['excitation_center_frequecy']
    output['excitation_bw'] = simulation['excitation_bandwidth']

    meta = dict()

    return output, meta