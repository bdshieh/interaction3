## mfield / simulations / beamplot_transmit_simulation.py

import numpy as np
import attr
from scipy.signal import gausspulse
from interaction3 import abstract
from .. core import mfield

@attr.s
class BeamplotTransmitSimulation(object):


    frequency = attr.ib()
    density = attr.ib()
    sound_speed = attr.ib()
    attenuation = attr.ib()
    frequency_attenuation = attr.ib()
    use_attenuation = attr.ib()
    use_element_factor = attr.ib()
    field_pos = attr.ib(default=None)

    excitation_center_frequecy = attr.ib()
    excitation_bandwidth = attr.ib()
    excitation_sample_frequency = attr.ib()

    rectangles = attr.ib()
    channel_centers = attr.ib()
    delays = attr.ib()
    apodization = attr.ib()

    _field = attr.ib(init=False, repr=False)
    _result = attr.ib(init=False, default=attr.Factory(dict), repr=False)

    @_field.default
    def _field_default(self):

        field = mfield.MField()
        field.field_init()
        field.set_field('c', self.sound_speed)
        field.set_field('fs', self.sample_frequency)
        field.set_field('att', self.attenuation)
        field.set_field('freq_att', self.frequency_attenuation)
        field.set_field('att_f0', self.attenuation_center_frequency)
        field.set_field('use_att', self.use_attenuation)

        return field


    def __attr_post_init__(self):

        field = self._field

        # create excitation signal
        fc = self.excitation_center_frequency
        fbw = self.excitation_bandwidth / fc
        fs = self.excitation_sample_frequency

        cutoff = gausspulse('cutoff', fc=fc, bw=fbw, tpr=-100, bwr=-3)
        adj_cutoff = np.ceil(cutoff * fs) / fs
        t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
        impulse_response, _ = gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

        # get transmit channel positions
        rect = self.rectangles
        cc = self.channel_centers
        delays = self.delays
        apod = self.apodization

        tx = field.xdc_rectangles(rect, cc, np.array([[0, 0, 300]]))
        field.xdc_impulse(tx, impulse_response)
        field.xdc_excitation(tx, np.array([1]))
        field.xdc_focus_times(tx, np.zeros((1, 1)), delays)
        field.xdc_apodization(tx, np.zeros((1, 1)), apod)

    def solve(self, field_pos=None):

        field = self._field

        if field_pos is None:
            field_pos = self.field_pos

        rf_data = list()
        t0s = list()
        image_data = np.zeros(len(field_pos), dtype=np.float64)

        for i, pos in enumerate(field_pos):

            if use_element_factor:
                # calculate element factor corrections
                r_tx = np.atleast_2d(pos) - np.atleast_2d(txpos)

                r, a, b = cart2sec(r_tx).T
                tx_correction = interpolator(np.rad2deg(np.abs(a)), np.rad2deg(np.abs(b)), fcubic)

                # apply correction as apodization
                field.xdc_apodization(tx, np.zeros((1, 1)), tx_apod * tx_correction)

            rf, t0 = field.calc_hp(tx, pos)

            rf_data.append(rf)
            t0s.append(t0)
            # image_data[i] = np.max(envelope(rf))

        return rf_data, t0s, image_data, idx


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
    '''
    Converts array/membrane definitions to a form that can be used in
    Field II.
    '''

    # create lists to store info about each mathematical element in the
    # array
    channel_no = list()
    vertices = list()
    apodizations = list()
    widths = list()
    heights = list()
    centers = list()

    channel_counter = 1
    channel_centers = list()
    delays = list()
    ch_apod = list()

    for ch in array['channels']:

        ch_center = ch['center']

        delay = ch.get('delay', 0)
        ch_apod = ch.get('apodization', 1)

        for elem in ch['elements']:

            elem_apod = elem.get('apodization', 1)

            for mem in elem['membranes']:

                # pull membrane info
                center = mem['center']
                length_x = mem['length_x']
                length_y = mem['length_y']

                ndiv_x = mem.get('ndiv_x', 2)
                ndiv_y = mem.get('ndiv_y', 2)
                mem_apod = mem.get('apodization')
                rotations = mem.get('rotations', None)

                elem_length_x = length_x / ndiv_x
                elem_length_y = length_y / ndiv_y

                # use meshgrid to determine centers of mathematical elements
                xv = np.linspace(-length_x / 2 + elem_length_x / 2, length_x / 2 - elem_length_x / 2, ndiv_x,
                                    endpoint=True)
                yv = np.linspace(-length_y / 2 + elem_length_y / 2, length_y / 2 - elem_length_y / 2, ndiv_y,
                                    endpoint=True)
                zv = 0
                x, y, z = np.meshgrid(xv, yv, zv, indexing='xy')

                # elem_centers = np.concatenate((x[..., None], y[..., None], np.zeros_like(x)[..., None]), axis=-1)
                math_elem_centers = np.c_[x.ravel(), y.ravel(), z.ravel()]

                if rotations is not None:
                    for vec, angle in rotations:
                        math_elem_centers = rotate_nodes(math_elem_centers, vec, angle)

                math_elem_centers += ch_center

                for i in range(ndiv_x):
                    for j in range(ndiv_y):

                        elem_center = elem_centers[j, i, :]

                        # determine vertices in clock-wise order starting from
                        # the lower left
                        vert_00 = np.array([-elem_length_x / 2, -elem_length_y / 2, 0])
                        vert_01 = np.array([-elem_length_x / 2, elem_length_y / 2, 0])
                        vert_11 = np.array([elem_length_x / 2, elem_length_y / 2, 0])
                        vert_10 = np.array([elem_length_x / 2, -elem_length_y / 2, 0])

                        if rotations is not None:
                            for vec, angle in rotations:
                                vert_00 = rotate_nodes(vert_00, vec, angle)
                                vert_01 = rotate_nodes(vert_01, vec, angle)
                                vert_11 = rotate_nodes(vert_11, vec, angle)
                                vert_10 = rotate_nodes(vert_10, vec, angle)

                        vert_00 += elem_center
                        vert_01 += elem_center
                        vert_11 += elem_center
                        vert_10 += elem_center

                        widths.append(elem_length_x)
                        channel_no.append(channel_counter)
                        heights.append(elem_length_y)
                        centers.append(elem_center)
                        vertices.append(np.concatenate((vert_00, vert_01, vert_11, vert_10), axis=0))
                        apodizations.append(apod)

            channel_counter += 1

    # convert lists to numpy arrays and concatenate
    a = np.array(channel_no)[..., None]
    b = np.array(vertices)
    c = np.array(apodizations)[..., None]
    d = np.array(widths)[..., None]
    e = np.array(heights)[..., None]
    f = np.array(centers)

    rect = np.concatenate((a, b, c, d, e, f), axis=1)

    channel_centers = np.array(channel_centers)
    delays = np.array(delays)
    ch_apod = np.array(ch_apod)

    kwargs = dict()
    kwargs['rect'] = rect
    kwargs['channel_centers'] = channel_centers
    kwargs['delays'] = delays
    kwargs['ch_apod'] = ch_apod

    meta = dict()

    return kwargs, meta


def rotation_matrix(vec, angle):

    x, y, z = vec
    a = angle

    r = np.zeros((3, 3))
    r[0, 0] = np.cos(a) + x**2 * (1 - np.cos(a))
    r[0, 1] = x * y * (1 - np.cos(a)) - z * np.sin(a)
    r[0, 2] = x * z * (1 - np.cos(a)) + y * np.sin(a)
    r[1, 0] = y * x * (1 - np.cos(a)) + z * np.sin(a)
    r[1, 1] = np.cos(a) + y**2 * (1 - np.cos(a))
    r[1, 2] = y * z * (1 - np.cos(a)) - x * np.sin(a)
    r[2, 0] = z * x * (1 - np.cos(a)) - z * np.sin(a)
    r[2, 1] = z * y * (1 - np.cos(a)) + x * np.sin(a)
    r[2, 2] = np.cos(a) + z**2 * (1 - np.cos(a))

    return r


def rotate_nodes(nodes, vec, angle):

    rmatrix = rotation_matrix(vec, angle)
    return rmatrix.dot(nodes.T).T