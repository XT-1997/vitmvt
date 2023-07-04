import os
import subprocess
import time
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 0
MINOR = 17
PATCH = '0'
SUFFIX = 'a0'
if PATCH:
    SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)
else:
    SHORT_VERSION = '{}.{}{}'.format(MAJOR, MINOR, SUFFIX)

version_file = 'vitmvt/version.py'


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from vitmvt.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}

__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha
    VERSION = os.getenv('VITMVTSDK_VERSION', VERSION)

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    if '--' in version:
                        # the `extras_require` doesn't accept options.
                        version = version.split('--')[0].strip()
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    write_version_py()
    s = setup(
        name='vitmvt',
        version=get_version(),
        description='Generalized Machine Learning Toolbox',
        packages=find_packages(exclude=('tools', 'demo')),
        zip_safe=False,
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        entry_points={
            'console_scripts': [
                'search-run = vitmvt.command.entry:run',
                'search-init = vitmvt.command.entry:init',
                'search-ctl = vitmvt.command.entry:control'
            ],
        },
    )
    try:
        installation_path = s.command_obj['install'].install_lib
        install_usersite_path = s.command_obj['install'].install_usersite + '/'
        for install_path in [installation_path, install_usersite_path]:
            for filename in os.listdir(install_path):
                if 'mme' in filename:
                    init_file_path = os.path.join(install_path, filename,
                                                  '__init__.py')
                    if not os.path.exists(init_file_path):
                        init_file_path = os.path.join(install_path, filename,
                                                      'mme/__init__.py')
                        if not os.path.exists(init_file_path):
                            continue
                    lines = [line for line in open(init_file_path)]
                    newlines = []
                    for line in lines:
                        if 'from .version' in line:
                            newlines += [line, '\n', '\n']
                        if '__all__' in line:
                            newlines.append(line)
                    with open(init_file_path, 'w') as f:
                        f.writelines(newlines)
                    print(init_file_path)
                    break
    except Exception as e:
        print(e)
