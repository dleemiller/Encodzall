import functools
import os
import toml


class ToObject(object):
    """
    https://stackoverflow.com/a/6993694
    """

    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return ToObject(value) if isinstance(value, dict) else value


def load_config(model_name):
    path = os.path.join(os.path.dirname(__file__), f"{model_name}.toml")
    with open(path, "r") as fh:
        config = toml.load(fh)
    return ToObject(config)


load_s = functools.partial(load_config, "encodzall_small")

__all__ = [load_config, load_s]
