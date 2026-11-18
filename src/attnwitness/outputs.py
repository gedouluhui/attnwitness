"""Protected same-directory publication. Not a multi-file transaction."""
import os
from pathlib import Path
import tempfile
from .contracts import ContractError


def _destination(path):
    p = Path(path).absolute()
    return p.parent.resolve() / p.name


def validate_outputs(paths, *, inputs=(), force=False):
    destinations = [_destination(p) for p in paths]
    sources = [Path(p).resolve() for p in inputs if p is not None]
    for index, destination in enumerate(destinations):
        if not destination.parent.is_dir():
            raise ContractError('output parent directory must already exist')
        exists = os.path.lexists(destination)
        if exists and not force:
            raise ContractError('output exists; use --force to replace a report')
        for other in sources + destinations[:index]:
            same = destination.resolve() == other.resolve()
            if not same and destination.exists() and other.exists():
                same = os.path.samefile(destination, other)
            if same:
                raise ContractError('output aliases an input or another output file')
    return destinations


def publish_reports(reports, *, inputs=(), force=False):
    """Stage complete UTF-8 contents before publishing any file.

No-force uses link()'s atomic create-if-absent behavior. Force replaces the leaf
entry instead of truncating its existing inode. This protects pre-existing hard
links and does not promise a transaction spanning multiple destination files.
Directory manipulation by a hostile concurrent actor is outside this contract.
"""
    destinations = validate_outputs([p for p, _ in reports], inputs=inputs, force=force)
    staged = []
    try:
        for destination, (_, text) in zip(destinations, reports):
            fd, name = tempfile.mkstemp(prefix='.attnwitness-', suffix='.tmp', dir=destination.parent)
            temporary = Path(name)
            staged.append((temporary, destination))
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            validate_outputs([destination], inputs=inputs, force=force)
            if force:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
                temporary.unlink()
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
