from pathlib import Path


def trace_up(target_file):
    dir = Path.cwd()
    if target_file == '.search-run':
        # .search-run must be in current directory or the whole path
        if not (dir / target_file).exists() and target_file not in str(dir):
            raise FileNotFoundError(
                '.search-run is not exists, please make sure it is in ' +
                'current directory or the current path.')
    while dir != Path('/') and not (dir / target_file).is_dir():
        dir = dir.parent
    path = dir / target_file
    if path.is_dir():
        return str(path)
    else:
        return None
