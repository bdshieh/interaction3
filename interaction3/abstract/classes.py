## interaction3 / abstract / classes.py

__all__ = ['SquareCmutMembrane', 'CircularCmutMembrane', 'SquarePmutMembrane', 'CircularPmutMembrane',
           'Channel', 'DefocusedChannel', 'Array', 'BemSimulation', 'MfieldSimulation']

from namedlist import namedlist, FACTORY
from collections import OrderedDict
from copy import deepcopy, copy
import sys, inspect
import json


# def __copy__(self):
#
#     return type(self)(_skip_validation=True, **self)
#
# def __deepcopy__(self, memo):
#
#     return type(self)(_skip_validation=True, **deepcopy(dict(self), memo=memo))

def _generate_object_with_type(obj):

    _type = type(obj).__name__

    if _type in classes or _type is 'dict':

        d = {}
        d['_type'] = _type
        for k, v in obj._asdict().items():
            d[k] = _generate_object_with_type(v)
        return d

    elif isinstance(obj, (list, tuple)):

        l = []
        for i in obj:
            l.append(_generate_object_with_type(i))
        return l

    else:
        return obj


def _generate_object_from_json(js):

    if isinstance(js, dict):

        _type = js.pop('_type')

        d = {}
        for key, val in js.items():
            d[key] = _generate_object_from_json(val)

        if _type in classes:
            return ObjectFactory.create(_type, **d)

        return d

    elif isinstance(js, (list, tuple)):

        l = []
        for i in js:
            l.append(_generate_object_from_json(i))
        return l

    else:
        return js


class ObjectFactory:

    @staticmethod
    def create(_type, *args, **kwargs):
        return globals()[_type](*args, **kwargs)


def _repr(self):
    return self.__str__()


def _str(self):
    return pretty(self)


def dump(obj, fp, indent=1, mode='x', *args, **kwargs):
    json.dump(_generate_object_with_type(obj), open(fp, mode), indent=indent, *args, **kwargs)


def dumps(obj, indent=1, *args, **kwargs):
    return json.dumps(_generate_object_with_type(obj), indent=indent, *args, **kwargs)


def load(fp, *args, **kwargs):
    return _generate_object_from_json(json.load(open(fp, 'r'), *args, **kwargs))


def loads(s, *args, **kwargs):
    return _generate_object_from_json(json.loads(s, *args, **kwargs))


def abstracttype(*args, **kwargs):

    cls = namedlist(*args, **kwargs)

    cls.__repr__ = _repr
    cls.__str__ = _str

    return cls


def pretty(obj, indent=0):

    strings = []

    if type(obj).__name__ in classes:

        # print(' ' * indent, type(obj).__name__, sep='')
        strings += [' ' * indent, type(obj).__name__, '\n']

        for key, val in obj._asdict().items():

            if type(val).__name__ in classes:
                # print(' ' * (indent + 1), str(key), ': ', sep='')
                strings += [' ' * (indent + 1), str(key), ': ', '\n']
                strings += [pretty(val, indent + 1)]

            elif isinstance(val, (list, tuple)):
                # print(' ' * (indent + 1), str(key), ': ', sep='')
                strings += [' ' * (indent + 1), str(key), ': ', '\n']
                strings += [pretty(val, indent + 2)]

            else:
                # print(' ' * (indent + 1), str(key), ': ', str(val), sep='')
                strings += [' ' * (indent + 1), str(key), ': ', str(val), '\n']

    elif isinstance(obj, (list, tuple)):

        if len(obj) == 0:
            # print(' ' * indent, '[]', sep='')
            strings += [' ' * indent , '[]', '\n']

        elif type(obj[0]).__name__ in classes:

            for val in obj:
                strings += [pretty(val, indent + 1)]

        elif isinstance(obj[0], (list, tuple)):

            for val in obj:
                strings += [pretty(val, indent + 1)]

        else:
            # print(' ' * indent, str(obj))
            strings += [' ' * indent, str(obj), '\n']
    else:
        pass

    return ''.join(strings)


_SquareCmutMembrane = OrderedDict()
_SquareCmutMembrane['id'] = None
_SquareCmutMembrane['length_x'] = 35e-6
_SquareCmutMembrane['length_y'] = 35e-6
_SquareCmutMembrane['electrode_x'] = 35e-6
_SquareCmutMembrane['electrode_y'] = 35e-6
_SquareCmutMembrane['thickness'] = (2e-6,)
_SquareCmutMembrane['density'] = (2040,)
_SquareCmutMembrane['y_modulus'] = (110e9,)
_SquareCmutMembrane['p_ratio'] = (0.22,)
_SquareCmutMembrane['isolation'] = 50e-9
_SquareCmutMembrane['permittivity'] = 6.3
_SquareCmutMembrane['gap'] = 100e-9
_SquareCmutMembrane['att_mech'] = 0
_SquareCmutMembrane['nnodes_x'] = 9
_SquareCmutMembrane['nnodes_y'] = 9
_SquareCmutMembrane['ndiv_x'] = 2
_SquareCmutMembrane['ndiv_y'] = 2

_CircularCmutMembrane = OrderedDict()
_CircularCmutMembrane['id'] = None
_CircularCmutMembrane['radius'] = 35e-6
_CircularCmutMembrane['electrode_r'] = 35e-6
_CircularCmutMembrane['thickness'] = (2e-6,)
_CircularCmutMembrane['density'] = (2040,)
_CircularCmutMembrane['y_modulus'] = (110e9,)
_CircularCmutMembrane['p_ratio'] = (0.22,)
_CircularCmutMembrane['isolation'] = 50e-9
_CircularCmutMembrane['permittivity'] = 6.3
_CircularCmutMembrane['gap'] = 100e-9
_CircularCmutMembrane['att_mech'] = 0
_CircularCmutMembrane['nnodes_x'] = 9
_CircularCmutMembrane['nnodes_y'] = 9
_CircularCmutMembrane['ndiv_x'] = 2
_CircularCmutMembrane['ndiv_y'] = 2

_SquarePmutMembrane = OrderedDict()
_CircularPmutMembrane = OrderedDict()

_Channel = OrderedDict()
_Channel['id'] = None
_Channel['position'] = None
_Channel['kind'] = None
_Channel['active'] = True
_Channel['apodization'] = 1
_Channel['delay'] = 0
_Channel['dc_bias'] = 0
_Channel['membranes'] = FACTORY(list)

_DefocusedChannel = OrderedDict()

_Array = OrderedDict()
_Array['id'] = None
_Array['position'] = None
_Array['delay'] = 0
_Array['rotation_origin'] = None
_Array['vertices'] = FACTORY(list)
_Array['channels'] = FACTORY(list)

_BemSimulation = OrderedDict()



_MfieldSimulation = OrderedDict()


SquareCmutMembrane = abstracttype('SquareCmutMembrane', _SquareCmutMembrane)
CircularCmutMembrane = abstracttype('CircularCmutMembrane', _CircularCmutMembrane)
SquarePmutMembrane = abstracttype('SquarePmutMembrane', _SquarePmutMembrane)
CircularPmutMembrane = abstracttype('CircularPmutMembrane', _CircularPmutMembrane)

Channel = abstracttype('Channel', _Channel)
DefocusedChannel = abstracttype('DefocusedChannel', _DefocusedChannel)

Array = abstracttype('Array', _Array)

BemSimulation = abstracttype('BemSimulation', _BemSimulation)
MfieldSimulation = abstracttype('MfieldSimulation', _MfieldSimulation)


classes = tuple(name for name, value in inspect.getmembers(sys.modules[__name__], inspect.isclass))


if __name__ == '__main__':

    a = Array(channels=[Channel(membranes=[SquareCmutMembrane()])] * 4)
    b = Array(channels=[Channel(membranes=SquareCmutMembrane())] * 4)
