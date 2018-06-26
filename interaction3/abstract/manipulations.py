## interaction3 / abstract / manipulations.py

# names to export
__all__ = ['move_membrane', 'translate_membrane', 'rotate_membrane', 'move_element', 'translate_element',
           'rotate_element', 'element_position_from_membranes', 'channel_position_from_elements', 'focus_channel',
           'defocus_channel', 'bias_channel', 'activate_channel', 'deactivate_channel', 'move_array',
           'translate_array', 'rotate_array', 'array_position_from_vertices', 'get_channel_positions_from_array',
           'get_element_positions_from_array', 'get_membrane_positions_from_array', 'focus_array']

import numpy as np
import math


## DECORATORS ##

def vectorize(f):

    def decorator(m, *args, **kwargs):

        if isinstance(m, (list, tuple)):
            res = list()
            for i in m:
                res.append(f(i, *args, **kwargs))
            return res
        else:

            return f(m, *args, **kwargs)
    return decorator


## HELPER FUNCTIONS ##

def rotation_matrix(vec, angle):

    if isinstance(vec, str):
        string = vec.lower()
        if string == 'x':
            vec = [1, 0, 0]
        elif string == '-x':
            vec = [-1, 0, 0]
        elif string == 'y':
            vec = [0, 1, 0]
        elif string == '-y':
            vec = [0, -1, 0]
        elif string == 'z':
            vec = [0, 0, 1]
        elif string == '-z':
            vec = [0, 0, -1]

    x, y, z = vec
    a = angle

    r = np.zeros((3, 3))
    r[0, 0] = np.cos(a) + x**2 * (1 - np.cos(a))
    r[0, 1] = x * y * (1 - np.cos(a)) - z * np.sin(a)
    r[0, 2] = x * z * (1 - np.cos(a)) + y * np.sin(a)
    r[1, 0] = y * x * (1 - np.cos(a)) + z * np.sin(a)
    r[1, 1] = np.cos(a) + y**2 * (1 - np.cos(a))
    r[1, 2] = y * z * (1 - np.cos(a)) - x * np.sin(a)
    r[2, 0] = z * x * (1 - np.cos(a)) - y * np.sin(a)
    r[2, 1] = z * y * (1 - np.cos(a)) + x * np.sin(a)
    r[2, 2] = np.cos(a) + z**2 * (1 - np.cos(a))

    return r


def distance(x, y):
    return math.sqrt(math.fsum([(i - j) ** 2 for i, j in zip(x, y)]))


## MEMBRANE MANIPLUATIONS ##

@vectorize
def move_membrane(m, pos):
    m['position'] = pos


@vectorize
def translate_membrane(m, vec):
    m['position'] = [i + j for i, j in zip(m['position'], vec)]


@vectorize
def rotate_membrane(m, origin, vec, angle):

    org = np.array(origin)
    pos = np.array(m['position'])

    # determine new membrane position
    newpos = rotation_matrix(vec, angle).dot(pos - org) + org
    m['position'] = newpos.tolist()

    # update membrane rotation list
    if 'rotations' in m:
        m['rotations'].append([vec, angle])
    else:
        m['rotations'] = [[vec, angle]]


## ELEMENT MANIPULATIONS ##

@vectorize
def move_element(e, pos):

    vec = [i - j for i, j in zip(pos, e['position'])]
    translate_element(e, vec)


@vectorize
def translate_element(e, vec):

    e['position'] = [i + j for i, j in zip(e['position'], vec)]

    for m in e['membranes']:
        translate_membrane(m, vec)


@vectorize
def rotate_element(e, origin, vec, angle):

    org = np.array(origin)
    pos = np.array(e['position'])

    # determine new element position
    newpos = rotation_matrix(vec, angle).dot(pos - org) + org
    e['position'] = newpos.tolist()

    # rotate membranes
    for m in e['membranes']:
        rotate_membrane(m, origin, vec, angle)


@vectorize
def element_position_from_membranes(e):

    membranes = e['membranes']

    x = [m['position'][0] for m in membranes]
    y = [m['position'][1] for m in membranes]
    z = [m['position'][2] for m in membranes]

    e['position'] = [np.mean(x), np.mean(y), np.mean(z)]


## CHANNEL MANIPULATIONS ##

@vectorize
def move_channel(ch, pos):

    vec = [i - j for i, j in zip(pos, ch['position'])]
    translate_channel(ch, vec)


@vectorize
def translate_channel(ch, vec):

    ch['position'] = [i + j for i, j in zip(ch['position'], vec)]

    for e in ch['elements']:
        translate_element(e, vec)


@vectorize
def rotate_channel(ch, origin, vec, angle):

    org = np.array(origin)
    pos = np.array(ch['position'])

    # determine new element position
    newpos = rotation_matrix(vec, angle).dot(pos - org) + org
    ch['position'] = newpos.tolist()

    # rotate membranes
    for e in ch['elements']:
        rotate_element(e, origin, vec, angle)


@vectorize
def channel_position_from_elements(ch):

    elements = ch['elements']

    x = [e['position'][0] for e in elements]
    y = [e['position'][1] for e in elements]
    z = [e['position'][2] for e in elements]

    ch['position'] = [np.mean(x), np.mean(y), np.mean(z)]


@vectorize
def focus_channel(ch, pos, sound_speed, quantization=None):

    d = distance(ch['position'], pos)
    if quantization is None or quantization == 0:
        t = d / sound_speed
    else:
        t = round(d / sound_speed / quantization) * quantization

    ch['delay'] = -t


@vectorize
def defocus_channel(ch, pos):
    raise NotImplementedError


@vectorize
def bias_channel(ch, bias):
    ch['dc_bias'] = bias


@vectorize
def activate_channel(ch):
    ch['active'] = True


@vectorize
def deactivate_channel(ch):
    ch['active'] = False


## ARRAY MANIPLUATIONS ##

@vectorize
def move_array(a, pos):

    vec = [i - j for i, j in zip(pos, a['position'])]
    translate_array(a, vec)


@vectorize
def translate_array(a, vec):

    a['position'] = [i + j for i, j in zip(a['position'], vec)]

    if 'vertices' in a:
        new_vertices = list()
        for v in a['vertices']:
            new_vertices.append([i + j for i, j in zip(v, vec)])
        a['vertices'] = new_vertices

    for ch in a['channels']:
        translate_channel(ch, vec)


@vectorize
def rotate_array(a, vec, angle, origin=None):

    if origin is None:
        origin = a['rotation_origin']

    org = np.array(origin)
    pos = np.array(a['position'])

    # determine new array position
    newpos = rotation_matrix(vec, angle).dot(pos - org) + org
    a['position'] = newpos.tolist()

    if 'vertices' in a:
        new_vertices = list()
        for v in a['vertices']:
            newpos = rotation_matrix(vec, angle).dot(np.array(v) - org) + org
            new_vertices.append(newpos.tolist())
        a['vertices'] = new_vertices

    # rotate channels
    for ch in a['channels']:
        rotate_channel(ch, origin, vec, angle)


@vectorize
def array_position_from_vertices(a):
    a['position'] = np.mean(np.array(a['vertices']), axis=0).tolist()


@vectorize
def get_channel_positions_from_array(a):
    return np.array([ch['position'] for ch in a['channels']])


@vectorize
def get_element_positions_from_array(a):
    return np.array([e['position'] for ch in a['channels'] for e in ch['elements']])


@vectorize
def get_membrane_positions_from_array(a):
    return np.array([m['position'] for ch in a['channels'] for e in ch['elements'] for m in e['membranes']])


@vectorize
def focus_array(a, pos, sound_speed, quantization=None, kind=None):

    if kind.lower() in ['tx', 'transmit']:
        channels = [ch for ch in a['channels'] if ch['kind'].lower() in ['tx', 'transmit', 'both', 'txrx']]
    elif kind.lower() in ['rx', 'receive']:
        channels = [ch for ch in a['channels'] if ch['kind'].lower() in ['rx', 'receive', 'both', 'txrx']]
    elif kind.lower() in ['txrx', 'both']:
        channels = a['channels']

    for ch in channels:
        focus_channel(ch, pos, sound_speed, quantization)


@vectorize
def reset_focus_array(a):
    for ch in a['channels']:
        ch['delay'] = 0


if __name__ == '__main__':

    pass


