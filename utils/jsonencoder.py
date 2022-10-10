
from _ctypes import PyObj_FromPtr
import json
import re

class NoIndent(object):
  FORMAT_SPEC = '@@{}@@'
  regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

  """ Value wrapper. """
  def __init__(self, value):
    self.value = value
  
  def __call__ (self, sort_keys):
    return json.dumps(self.value, cls = JsonEncoder, sort_keys = sort_keys)

class RightAlign(object):
  FORMAT_SPEC = '!!{}!!'
  regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

  """ Value wrapper. """
  def __init__(self, value, space):
    self.value = value
    self.formatter = '{:' + str(space) + '}'
    self.space = space

  def __call__ (self, sort_keys):
    return self.formatter.format(self.value)

_wrappers = [
  NoIndent,
  RightAlign
]

class JsonEncoder(json.JSONEncoder):

  def __init__(self, **kwargs):
    # Save copy of any keyword argument values needed for use here.
    self.__sort_keys = kwargs.get('sort_keys', None)
    self._ids = []
    super(JsonEncoder, self).__init__(**kwargs)

  def default(self, obj):
    for wrapper in _wrappers:
      if isinstance(obj, wrapper):
        id_obj = id(obj)
        self._ids.append(id_obj)
        return wrapper.FORMAT_SPEC.format(id_obj)
    return super(JsonEncoder, self).default(obj)

  def encode(self, obj):
    json_repr = super(JsonEncoder, self).encode(obj)  # Default JSON.

    for wrapper in _wrappers:
      format_spec = wrapper.FORMAT_SPEC  # Local var to expedite access.

      # Replace any marked-up object ids in the JSON repr with the
      # value returned from the json.dumps() of the corresponding
      # wrapped Python object.
      for match in wrapper.regex.finditer(json_repr):
        # see https://stackoverflow.com/a/15012814/355230
        id = int(match.group(1))
        if id in self._ids:
          wrapper_obj = PyObj_FromPtr(id)

          # Replace the matched id string with json formatted representation
          # of the corresponding Python object.
          json_repr = json_repr.replace('"{}"'.format(format_spec.format(id)), wrapper_obj(self.__sort_keys))
    return json_repr
