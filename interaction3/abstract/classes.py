
import json
import jsonschema
import os
import re

SCHEMA_FILENAME = 'base-schema-1.0.json' # relative path to schema json file


## BASE CLASSES ##

class BaseList(list):

    _name = None
    _class_reference = None

    def __init__(self, *args, **kwargs):

        super()

        # check if name is specified
        if '_name' in kwargs:
            self._name = kwargs['_name']

        object_name = self._name

        # base class not intended to be instantiated
        if object_name is None:
            raise Exception

        # set class reference for lookup and validation
        self._class_reference = BASE_REFERENCE[object_name]

        # set items which will be partially validated
        for index, value in enumerate(list(*args)):
            self.append(value)

        # validate entire object
        self.validate()

    def __repr__(self):
        return self.__str__()

    def __str__(self, indent=0):

        strings = list()

        for val in self:

            if isinstance(val, (BaseList, BaseDict)):
                strings.append(val.__str__(indent))
            else:
                strings.append(' ' * indent + str(val) + '\n')

        return ''.join(strings)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __setitem__(self, index, value):

        val = self._get_object_from_value(index, value)

        self.partial_validate(index, value)

        super().__setitem__(index, val)

    def append(self, value):

        index = len(self)
        val = self._get_object_from_value(index, value)

        self.partial_validate(index, value)

        super().append(val)

    def extend(self, iterable):
        super().extend(iterable)

    def insert(self, index, value):
        super().insert(index, value)

    def _get_item_name(self, index=None):

        # check if object at (optional) index has name property, otherwise return None
        class_reference = self._class_reference
        items = class_reference['items']

        # if items is an object, schema applies to entire list
        if isinstance(items, dict):
            if 'name' in items:
                return items['name']

        # if items is an array, only use schema at index
        elif isinstance(items, list):
            if 'name' in items[index]:
                return items[index]['name']

        return

    def _get_object_from_value(self, index, value):

        # if value is already object, do nothing
        if isinstance(value, (BaseList, BaseDict)):
            return value

        name = self._get_item_name(index)

        if name is not None:
            return ObjectFactory.create(name, value)

        return value

    def validate(self):

        class_reference = self._class_reference
        jsonschema.validate(self, class_reference)

    def partial_validate(self, index, value):

        class_reference = self._class_reference
        items = class_reference['items']

        # if items is an object, schema applies to entire list
        if isinstance(items, dict):
            schema = items

        # if items is an array, only use schema at index
        elif isinstance(items, list):
            schema = items[index]

        jsonschema.validate(value, schema)


class BaseDict(dict):

    _name = None
    _class_reference = None

    def __init__(self, *args, **kwargs):

        super()

        # check if name is specified
        if '_name' in kwargs:
            self._name = kwargs['_name']

        object_name = self._name

        # base class not intended to be instantiated
        if object_name is None:
            raise Exception

        # set class reference for lookup and validation
        self._class_reference = BASE_REFERENCE[object_name]

        # set key/value pairs which will be partially validated
        d = dict(*args, **kwargs)
        for key, val in d.items():
            self.__setitem__(key, val)

        # validate entire object
        self.validate()

    def __repr__(self):
        return self.__str__()

    def __str__(self, indent=0):

        strings = list()
        strings.append(' ' * indent + string_to_class_name(self._name) + '\n')

        for idx, (key, val) in enumerate(self.items()):

            strings.append(' ' * (indent + 2) + str(key) + ': ')

            if isinstance(val, (BaseList, BaseDict)):
                strings.append('\n' + val.__str__(indent + 4))
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

        # if value is already object, do nothing
        if isinstance(value, (BaseList, BaseDict)):
            return value

        name = self._get_attribute_name(key)

        if name is not None:
            return ObjectFactory.create(name, value)

        return value

    def validate(self):

        class_reference = self._class_reference
        jsonschema.validate(self, class_reference)

    def partial_validate(self, key, value):

        class_reference = self._class_reference
        schema = class_reference['properties'][key]
        jsonschema.validate(value, schema)

    # def __copy__(self):
        # pass


class ObjectFactory(object):

    @staticmethod
    def create(name, *args, **kwargs):

        kwargs['_name'] = name

        if name in ARRAYS:
            return BaseList(*args, **kwargs)
        elif name in OBJECTS:
            return BaseDict(*args, **kwargs)


## JSON HELPER FUNCTIONS ##

def get_base_schema(path):
    return json.load(open(path,'r'))


def resolve_json_refs(var):
    ''''''
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
    ''''''
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
    arrays = []

    for key, val in ref.items():
        if 'name' in val:
            if val['type'] == 'object':
                objects.append(key)
            elif val['type'] == 'array':
                arrays.append(key)

    return objects, arrays


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

    elif isinstance(var, (BaseList, list)):

        l = list()

        if isinstance(var, BaseList):
            l.append(var._name)

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

        if isinstance(l[0], str):

            object_name = l.pop(0)
            return ObjectFactory.create(object_name, l)

        return l

    else:
        return var


def dump(obj, fp, indent=2, *args, **kwargs):
    json.dump(generate_json_object_with_name(obj), open(fp, 'x'), indent=indent, *args, **kwargs)


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
OBJECTS, ARRAYS = get_classes_from_reference(BASE_REFERENCE)


## CLASS DEFINITIONS ##

class Membrane(BaseDict):
    _name = 'membrane'


class SquareCmutMembrane(BaseDict):
    _name = 'square_cmut_membrane'


class CircularCmutMembrane(BaseDict):
    _name = 'circular_cmut_membrane'


class SquarePmutMembrane(BaseDict):
    _name = 'square_pmut_membrane'


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


class Membranes(BaseList):
    _name = 'membranes'


class Elements(BaseList):
    _name = 'elements'


class Channels(BaseList):
    _name = 'channels'


if __name__ == '__main__':

    pass