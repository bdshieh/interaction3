## interaction3 / abstract / classes.py

# names to export
__all__ = ['SquareCmutMembrane', 'CircularCmutMembrane', 'SquarePmutMembrane', 'CircularPmutMembrane',
           'Channel', 'DefocusedChannel', 'Element', 'Array', 'Simulation', 'BemTransmitCrosstalk',
           'BemReceiveCrosstalk', 'BemTransmitBeamplot', 'MfieldTransmitBeamplot', 'dump', 'dumps', 'load', 'loads',
           'MfieldSimulation', 'BemSimulation']

import json
import jsonschema
import os
import re
from copy import deepcopy, copy
import functools

SCHEMA_FILENAME = 'base-schema-1.3.json' # relative path to schema json file


# def memoize(f):
#     '''
#     Class memoization using json serialization as the key and deepcopy.
#     '''
#     memo = dict()
#
#     @functools.wraps(f)
#     def decorator(*args, **kwargs):
#
#         key = json.dumps((args, kwargs))
#
#         if key not in memo:
#             memo[key] = f(*args, **kwargs)
#         return deepcopy(memo[key])
#
#     return decorator


def memoize(cls):
    '''
    Class memoization using json serialization as the key and deepcopy.
    '''
    memo = dict()

    class decorator(cls):

        def __call__(self, *args, **kwargs):

            key = json.dumps((args, kwargs))

            if key not in memo:
                memo[key] = cls.__call__(*args, **kwargs)

            return deepcopy(memo[key])

    return decorator


## BASE CLASSES ##

class BaseList(list):
    '''
    Base class for Lists.
    '''
    _name = None
    depth = 1 # default depth for __str__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return self.__str__()

    def __str__(self, indent=0, depth=None):

        if depth is None:
            depth = self.depth

        if depth <= 0: # if max depth is reached
            return '[...]' + '\n'
        elif len(self) == 0: # if empty
            return '[]' + '\n'
        elif isinstance(self[0], (BaseList, BaseDict)): # if list contains BaseList or BaseDict

            strings = list('\n')
            for val in self:
                strings.append(val.__str__(indent, depth=depth - 1))

            return ''.join(strings)
        else: # if list contains anything else (int, str, etc.)
            return str(list(self)) + '\n'

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __setitem__(self, index, value):

        val = self._get_object_from_value(value)
        super().__setitem__(index, val)

    def append(self, value):

        index = len(self)
        val = self._get_object_from_value(value)
        super().append(val)

    def extend(self, iterable):
        super().extend(iterable)

    def insert(self, index, value):
        super().insert(index, value)

    def _get_object_from_value(self, value):

        if isinstance(value, (BaseList, BaseDict)): # if value is already object, do nothing
            return value
        elif isinstance(value, (list, tuple)): # convert lists and tuples
            return BaseList(value)
        else:
            return value # do nothing

    # def __getstate__(self):
    #     return dumps(self)
    #
    # def __setstate__(self, state):
    #     self.__dict__ = loads(state).__dict__

class BaseDict(dict):
    '''
    Base class for Objects.
    '''
    _name = None
    _class_reference = None
    _validator = None
    depth = 1 # default depth for __str__

    def __init__(self, *args, **kwargs):

        super()

        # check if name is specified
        if '_name' in kwargs:
            self._name = kwargs.pop('_name')
        object_name = self._name

        # base class not intended to be instantiated
        if object_name is None:
            raise Exception

        # set class reference for lookup and validation
        self._class_reference = BASE_REFERENCE[object_name]
        self._validator = jsonschema.Draft4Validator(self._class_reference)

        # option to skip validation for faster init
        _skip_validation = kwargs.pop('_skip_validation', False)

        if _skip_validation:
            # set key/value pairs without any validation
            d = dict(*args, **kwargs)
            for key, val in d.items():
                super().__setitem__(key, val)

        else:
            # set key/value pairs which will be partially validated
            d = dict(*args, **kwargs)
            for key, val in d.items():
                self.__setitem__(key, val)

            # validate entire object (to make sure required attributes are set)
            self.validate()

    def __repr__(self):
        return self.__str__()

    def __str__(self, indent=0, depth=None):

        if depth is None:
            depth = self.depth

        strings = list()
        strings.append(' ' * indent + string_to_class_name(self._name) + '\n') # header with object name

        # determine string representation of each key, value pair
        for idx, (key, val) in enumerate(self.items()):

            strings.append(' ' * (indent + 2) + str(key) + ': ')

            if isinstance(val, BaseList) and depth == 0:
                strings.append(val.__str__(indent + 4, depth=depth))
            elif isinstance(val, (BaseList, BaseDict)): # ? this needs to be fixed for rare case of dict in dict
                strings.append(val.__str__(indent + 4, depth=depth))
            else:
                strings.append(str(val) + '\n')

        return ''.join(strings)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value):

        # only set items for valid keys
        if key not in self._get_valid_attributes():
            return

        # check if value should be converted to an object
        val = self._get_object_from_value(key, value)

        # validate key/value pair (partial validation)
        self.partial_validate(key, value)

        # set item
        super().__setitem__(key, val)

    def _get_valid_attributes(self):

        # valid attributes defined by reference
        class_reference = self._class_reference
        return class_reference['properties'].keys()

    def _get_attribute_name(self, key):

        # check if attribute has name property, otherwise return None
        class_reference = self._class_reference
        if 'name' in class_reference['properties'][key]:
            return class_reference['properties'][key]['name']

        return

    def _get_object_from_value(self, key, value):

        # if value is already BaseList or BaseDict, do nothing
        if isinstance(value, (BaseList, BaseDict)):
            return value

        # if value is python list, turn into a BaseList
        if isinstance(value, (list, tuple)):
            return BaseList(value)

        # try to create object from value
        name = self._get_attribute_name(key)
        if name is not None:
            return ObjectFactory.create(name, value)

        # if none of the above, return value unchanged
        return value

    def validate(self):
        self._validator.validate(self)

    def partial_validate(self, key, value):

        class_reference = self._class_reference
        schema = class_reference['properties'][key]
        jsonschema.Draft4Validator(schema).validate(value)

    def __copy__(self):
        # skip validation in init to speed up shallow copy
        return type(self)(_skip_validation=True, **self)

    def __deepcopy__(self, memo):
        # skip validation in init to speed up deep copy
        return type(self)(_skip_validation=True, **deepcopy(dict(self), memo=memo))

    # def __getstate__(self):
    #     return dumps(self, indent=0)
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(loads(state).__dict__)

class ObjectFactory(object):

    @staticmethod
    def create(name, *args, **kwargs):

        kwargs['_name'] = name
        return globals()[string_to_class_name(name)](*args, **kwargs)


## JSON HELPER FUNCTIONS ##

def get_base_schema(path):
    return json.load(open(path,'r'))


def resolve_json_refs(var):
    '''
    '''
    if type(var) is dict:

        d = {}
        for key, val in var.items():

            if key == '$ref':

                path, ref = RESOLVER.resolve(val)
                d = resolve_json_refs(ref)

            else:
                d[key] = resolve_json_refs(val)

        return d

    elif type(var) is list:

        l = []
        for i in var:
            l.append(resolve_json_refs(i))

        return l

    else:
        return var


def parse_json_allof(var):
    '''
    '''
    if isinstance(var, dict):

        d = {}
        for key, val in var.items():

            if key == 'allOf':

                first_obj = parse_json_allof(val[0])

                for rem_obj in val[1:]:

                    for k, v in rem_obj.items():

                        if k in first_obj:

                            if type(v) is dict:
                                first_obj[k].update(parse_json_allof(v))
                            elif type(v) is list:
                                first_obj[k] += parse_json_allof(v)
                            else:
                                raise Exception
                        else:
                            first_obj[k] = parse_json_allof(v)

                for k, v in first_obj.items():
                    if k not in d:
                        d[k] = v

            else:
                d[key] = parse_json_allof(val)

        return d

    elif isinstance(var, list):

        l = []
        for i in var:
            l.append(parse_json_allof(i))

        return l

    else:
        return var


def get_classes_from_reference(ref):

    objects = []

    for key, val in ref.items():
        if 'name' in val:
            if val['type'] == 'object':
                objects.append(key)

    return objects


def string_to_class_name(string):

    # capitalize first letter
    string = re.sub(r'[A-Za-z]', lambda m: m.group().title(), string, count=1)

    # replace `*_<c>` with `*<C>` E.g., `Error_x` --> `ErrorX`
    string = re.sub(r'_[A-Za-z0-9]+', lambda m: m.group()[1:].title(), string)

    return str(string)


## (DE)SERIALIZATION FUNCTIONS ##

def generate_json_object_with_name(var):

    if isinstance(var, (BaseDict, dict)):

        d = dict()

        if isinstance(var, BaseDict):
            d['_name'] = var._name

        for key, val in var.items():
            d[key] = generate_json_object_with_name(val)

        return d

    elif isinstance(var, (BaseList, list, tuple)):

        l = list()

        # if isinstance(var, BaseList):
            # l.append(var._name)

        for i in var:
            l.append(generate_json_object_with_name(i))

        return l

    else:
        return var


def generate_objects_from_json(var):

    if isinstance(var, dict):

        d = dict()

        for key, val in var.items():

            d[key] = generate_objects_from_json(val)

        if '_name' in d:

            object_name = d.pop('_name')
            return ObjectFactory.create(object_name, d)

        return d

    elif isinstance(var, list):

        l = []
        for i in var:
            l.append(generate_objects_from_json(i))

        # if isinstance(l[0], str):

            # object_name = l.pop(0)
            # return ObjectFactory.create(object_name, l)

        return l

    else:
        return var


def dump(obj, fp, indent=2, mode='x', *args, **kwargs):
    json.dump(generate_json_object_with_name(obj), open(fp, mode), indent=indent, *args, **kwargs)


def dumps(obj, indent=2, *args, **kwargs):
    return json.dumps(generate_json_object_with_name(obj), indent=indent, *args, **kwargs)


def load(fp, *args, **kwargs):
    return generate_objects_from_json(json.load(open(fp, 'r'), *args, **kwargs))


def loads(s, *args, **kwargs):
    return generate_objects_from_json(json.loads(s, *args, **kwargs))


## SETUP GLOBALS ##

base_dir = os.path.abspath(os.path.dirname(__file__))
schema_path = os.path.join(base_dir, SCHEMA_FILENAME)
schema_resolver_path = 'file://' + schema_path

BASE_SCHEMA = get_base_schema(schema_path)
RESOLVER = jsonschema.RefResolver(schema_resolver_path, BASE_SCHEMA)
BASE_REFERENCE = parse_json_allof(resolve_json_refs(BASE_SCHEMA))
OBJECTS = get_classes_from_reference(BASE_REFERENCE)


## CLASS DEFINITIONS ##
# Membrane classes are memoized to speed up their creation since they are instantiated frequently

@memoize
class SquareCmutMembrane(BaseDict):
    _name = 'square_cmut_membrane'


@memoize
class CircularCmutMembrane(BaseDict):
    _name = 'circular_cmut_membrane'


@memoize
class SquarePmutMembrane(BaseDict):
    _name = 'square_pmut_membrane'


@memoize
class CircularPmutMembrane(BaseDict):
    _name = 'circular_pmut_membrane'


class Element(BaseDict):
    _name = 'element'


class Array(BaseDict):
    _name = 'array'


class Channel(BaseDict):
    _name = 'channel'


class DefocusedChannel(BaseDict):
    _name = 'defocused_channel'


class Simulation(BaseDict):
    _name = 'simulation'


class BemTransmitCrosstalk(BaseDict):
    _name = 'bem_transmit_crosstalk'


class BemReceiveCrosstalk(BaseDict):
    _name = 'bem_receive_crosstalk'


class BemTransmitBeamplot(BaseDict):
    _name = 'bem_transmit_beamplot'


class BemReceiveBeamplot(BaseDict):
    _name = 'bem_receive_beamplot'


class MfieldTransmitBeamplot(BaseDict):
    _name = 'mfield_transmit_beamplot'


class MfieldTransmitReceiveBeamplotWithFoldingError(BaseDict):
    _name = 'mfield_transmit_receive_beamplot_with_folding_error'


class MfieldSimulation(BaseDict):
    _name = 'mfield_simulation'


class BemSimulation(BaseDict):
    _name = 'bem_simulation'


if __name__ == '__main__':

    pass