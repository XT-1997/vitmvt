import os.path as osp
import shutil
import sys
import tempfile
from importlib import import_module

from mmcv.fileio import BaseFileHandler, register_handler


def dict_to_text(cfg_dict):

    indent = 4

    def _indent(s_, num_spaces):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def _format_basic_types(k, v):
        if isinstance(v, str):
            v_str = "'{}'".format(v)
        else:
            v_str = str(v)
        attr_str = '{}={}'.format(str(k), v_str)
        attr_str = _indent(attr_str, indent)

        return attr_str

    def _format_list(k, v):
        # check if all items in the list are dict
        if all(isinstance(_, dict) for _ in v):
            v_str = '[\n'
            v_str += '\n'.join(
                'dict({}),'.format(_indent(_format_dict(v_), indent))
                for v_ in v).rstrip(',')
            attr_str = '{}={}'.format(str(k), v_str)
            attr_str = _indent(attr_str, indent) + ']'
        else:
            attr_str = _format_basic_types(k, v)
        return attr_str

    def _format_dict(d, outest_level=False):
        r = ''
        s = []
        for idx, (k, v) in enumerate(d.items()):
            is_last = idx >= len(d) - 1
            end = '' if outest_level or is_last else ','
            if isinstance(v, dict):
                v_str = '\n' + _format_dict(v)
                attr_str = '{}=dict({}'.format(str(k), v_str)
                attr_str = _indent(attr_str, indent) + ')' + end
            elif isinstance(v, list):
                attr_str = _format_list(k, v) + end
            else:
                attr_str = _format_basic_types(k, v) + end

            s.append(attr_str)
        r += '\n'.join(s)
        return r

    text = _format_dict(cfg_dict, outest_level=True)

    return text


@register_handler(file_formats='py')
class PythonHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        with tempfile.TemporaryDirectory() as temp_config_dir:
            filepath = osp.join(temp_config_dir, '_tempconfig.py')
            with open(filepath, 'w') as f:
                f.write(file.getvalue())
            sys.path.insert(0, temp_config_dir)
            mod = import_module('_tempconfig')
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            # delete imported module
            del sys.modules['_tempconfig']
        return cfg_dict

    def dump_to_fileobj(self, obj, file, **kwargs):
        file.write(self.dump_to_str(obj), **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return dict_to_text(obj)

    def load_from_path(self, filepath, mode='r', **kwargs):
        with tempfile.TemporaryDirectory() as temp_config_dir:
            shutil.copyfile(filepath,
                            osp.join(temp_config_dir, '_tempconfig.py'))
            sys.path.insert(0, temp_config_dir)
            mod = import_module('_tempconfig')
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            # delete imported module
            del sys.modules['_tempconfig']
        return cfg_dict

    def dump_to_path(self, obj, filepath, mode='w', **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
