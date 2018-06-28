## interaction3 / abstract / classes.py

__all__ = ['SquareCmutMembrane', 'CircularCmutMembrane', 'SquarePmutMembrane', 'CircularPmutMembrane',
           'Channel', 'DefocusedChannel', 'Array', 'BemSimulation', 'MfieldSimulation']

from namedlist import namedlist, FACTORY
from collections import OrderedDict
from copy import deepcopy, copy
import sys, inspect


# def __copy__(self):
#
#     return type(self)(_skip_validation=True, **self)
#
# def __deepcopy__(self, memo):
#
#     return type(self)(_skip_validation=True, **deepcopy(dict(self), memo=memo))


def _repr(self):
    return self.__str__()


def _str(self):
    return pretty(self)


def _dump(self):
    pass


def _dumps(self):
    pass


def _load(self):
    pass


def _loads(self):
    pass


def abstracttype(*args, **kwargs):

    cls = namedlist(*args, **kwargs)

    cls.__repr__ = _repr
    cls.__str__ = _str

    return cls


# def pprint(obj, indent=0, depth=0, maxdepth=2):
#
#     if type(obj).__name__ in classes:
#
#         if depth >= maxdepth:
#             if 'id' in obj._fields:
#                 return type(obj).__name__ + ' (id=' + str(obj.id) + ')' + '\n'
#             else:
#                 return type(obj).__name__ + '\n'
#
#         strings = []
#         strings.append(' ' * indent + type(obj).__name__ + '\n')
#
#         for key, val in obj._asdict().items():
#             strings.append(' ' * (indent + 2) + str(key) + ': ')
#
#             if type(val).__name__ in classes:
#                 strings.append('\n' + pprint(val, indent + 4, depth + 1, maxdepth))
#             else:
#                 strings.append(pprint(val, indent + 4, depth + 1, maxdepth))
#
#         return ''.join(strings)
#
#     elif isinstance(obj, (list, tuple)):
#
#         if len(obj) == 0:
#             return '[]' + '\n'
#
#         elif depth >= maxdepth:
#             return '[...]' + '\n'
#
#         elif type(obj[0]).__name__ in classes:
#
#             if depth == 0:
#                 strings = []
#             else:
#                 strings = ['\n', ]
#
#             for val in obj:
#                 strings.append(' ' * indent)
#                 strings.append(pprint(val, indent, depth + 1, maxdepth))
#             return ''.join(strings)
#
#         else:
#             return str(list(obj)) + '\n'
#
#     else:
#         return str(obj) + '\n'


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
