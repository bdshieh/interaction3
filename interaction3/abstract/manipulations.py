## interaction3 / abstract / manipulations.py

# names to export
__all__ = ['move_membrane', 'translate_membrane', 'rotate_membrane', 'move_element', 'translate_element',
           'rotate_element', 'element_position_from_membranes', 'channel_position_from_elements', 'focus_channel',
           'defocus_channel', 'bias_channel', 'activate_channel', 'deactivate_channel', 'move_array',
           'translate_array', 'rotate_array']

import numpy as np


## DECORATORS ##

def vectorize(f):

    def vf(m, *args, **kwargs):

        if isinstance(m, list):
            for i in m:
                f(i, *args, **kwargs)

        else: return f(m, *args, **kwargs)

    return vf


## ROTATION MATH ##

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
    m['rotations'].append([vec, angle])


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
def channel_position_from_elements(ch):

    elements = ch['elements']

    x = [e['position'][0] for e in elements]
    y = [e['position'][1] for e in elements]
    z = [e['position'][2] for e in elements]

    ch['position'] = [np.mean(x), np.mean(y), np.mean(z)]


@vectorize
def focus_channel(ch, pos):
    raise NotImplementedError


@vectorize
def defocus_channel(ch, pos):
    raise NotImplementedError


@vectorize
def bias_channel(ch, bias):
    ch['bias'] = bias


@vectorize
def activate_channel(ch):
    ch['active'] = True


@vectorize
def deactivate_channel(ch):
    ch['active'] = False


## ARRAY MANIPLUATIONS ##

@vectorize
def move_array(a, pos):

    for ch in a['channels']:
        move_ch


@vectorize
def translate_array(a, vec):
    raise NotImplementedError


@vectorize
def rotate_array(a, origin, vec, angle):
    raise NotImplementedError


if __name__ == '__main__':

    m1 = dict()
    m2 = dict()
    m1['position'] = [1,0,0]
    m2['position'] = [0,2,0]

    e = dict()
    e['membranes'] = [m1, m2]

    element_position_from_membranes(e)


