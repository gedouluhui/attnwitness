"""Offline CLI with explicit, bounded inputs and non-destructive outputs."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import numpy as np
from .audit import (audit, bounded_int, TILE_SIZE_RANGE, MAX_EVIDENCE_RANGE,
                    WINDOW_RADIUS_RANGE, MAX_FINDINGS_RANGE)
from .contracts import ContractError, Policy, Token
from .report import render_report
from .outputs import validate_outputs, publish_reports


def demo(case, **options):
    if case == 'prefix':
        tokens = [Token('prefix-0', 'doc', 0, prefix=True), Token('prefix-1', 'doc', 1, prefix=True),
                  Token('completion', 'doc', 2)]
        mask = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
        policy = Policy(tokens, mode='prefix')
    elif case == 'packed':
        policy = Policy([Token('document-A', 'A', 0), Token('document-B', 'B', 0)], mode='packed')
        mask = np.array([[True, False], [True, True]])
    else:
        policy = Policy([Token('first', 'doc', 0), Token('second', 'doc', 1)])
        mask = np.array([[True, False], [True, True]])
        if case == 'float-mask':
            mask = mask.astype(np.float32)
    result = audit(mask, policy, **options).to_dict()
    result['source'] = {'kind': 'bundled synthetic example', 'case': case}
    return result


def _options(args):
    bounded_int(args.tile_size, 'tile_size', *TILE_SIZE_RANGE)
    bounded_int(args.max_evidence, 'max_evidence', *MAX_EVIDENCE_RANGE)
    bounded_int(args.window_radius, 'window_radius', *WINDOW_RADIUS_RANGE)
    bounded_int(args.max_findings, 'max_findings', *MAX_FINDINGS_RANGE)
    return {'tile_size': args.tile_size, 'max_evidence': args.max_evidence,
            'window_radius': args.window_radius, 'max_findings': args.max_findings}


def _load_spec(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ContractError('contract JSON exceeds 64 KiB')
    def pairs(items):
        d = {}
        for key, value in items:
            if key in d:
                raise ContractError('duplicate JSON key')
            d[key] = value
        return d
    spec = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ContractError('nonfinite JSON value')))
    allowed = {'tokens', 'mode', 'require_nonempty', 'backend', 'batch_size', 'heads', 'is_causal'}
    if type(spec) is not dict or 'tokens' not in spec or set(spec) - allowed:
        raise ContractError('contract JSON has missing tokens or unsupported keys')
    tokens = spec['tokens']
    if type(tokens) is not list or not tokens:
        raise ContractError('tokens must be a nonempty JSON array')
    def batch(rows):
        if type(rows) is not list or any(type(row) is not dict for row in rows):
            raise ContractError('tokens must contain objects or batches of objects')
        return [Token(**row) for row in rows]
    parsed = batch(tokens) if type(tokens[0]) is dict else [batch(rows) for rows in tokens]
    policy = Policy(parsed, mode=spec.get('mode', 'causal'), require_nonempty=spec.get('require_nonempty', True))
    kwargs = {k: spec[k] for k in ('backend', 'batch_size', 'heads', 'is_causal') if k in spec}
    return policy, kwargs, hashlib.sha256(raw).hexdigest()


def _load_mask(path, max_cells, label='mask'):
    path = Path(path)
    byte_limit = max_cells * 8 + 4224
    with path.open('rb') as source:
        raw = source.read(byte_limit + 1)
    if len(raw) > byte_limit:
        raise ContractError(f'{label} file exceeds its byte budget')
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
    else:
        raise ContractError('only numeric NPY format 1.0/2.0 inputs are supported')
    if dtype.kind not in 'bf' or dtype.itemsize > 8 or not 2 <= len(shape) <= 4:
        raise ContractError(f'{label} NPY must contain a 2D..4D boolean/float array, never objects')
    if any(type(d) is not int or d < 1 for d in shape) or math.prod(shape) > max_cells:
        raise ContractError(f'{label} NPY exceeds cell budget or has empty dimensions')
    expected_size = stream.tell() + math.prod(shape) * dtype.itemsize
    if len(raw) != expected_size:
        raise ContractError(f'{label} NPY byte length does not match its header')
    stream.seek(0)
    mask = np.load(stream, allow_pickle=False, max_header_size=4096)
    return mask, hashlib.sha256(raw).hexdigest()


def _outputs(args, inputs=()):
    validate_outputs([p for p in (args.json, args.html) if p], inputs=inputs, force=args.force)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Audit declared attention visibility, offline.')
    sub = parser.add_subparsers(dest='command', required=True)
    demo_parser = sub.add_parser('demo', help='run a bundled correct or broken attention case')
    demo_parser.add_argument('--case', choices=['causal', 'float-mask', 'packed', 'prefix'], default='float-mask')
    check = sub.add_parser('check', help='audit explicit contract JSON and optional numeric NPY mask')
    check.add_argument('spec'); check.add_argument('mask', nargs='?')
    check.add_argument('--key-padding', help='optional MHA key-padding mask NPY of shape (B, L)')
    check.add_argument('--max-cells', type=int, default=1048576)
    for command in (demo_parser, check):
        command.add_argument('--json'); command.add_argument('--html')
        command.add_argument('--force', action='store_true')
        command.add_argument('--tile-size', type=int, default=128)
        command.add_argument('--max-evidence', type=int, default=16)
        command.add_argument('--window-radius', type=int, default=3)
        command.add_argument('--max-findings', type=int, default=128)
    args = parser.parse_args(argv)
    try:
        options = _options(args)
        inputs = (getattr(args, 'spec', None), getattr(args, 'mask', None),
                  getattr(args, 'key_padding', None))
        _outputs(args, inputs)
        if args.command == 'demo':
            result = demo(args.case, **options)
        else:
            if not 1 <= args.max_cells <= 16777216:
                raise ContractError('max-cells must be in 1..16777216 for CLI file loading')
            policy, kwargs, contract_sha = _load_spec(args.spec)
            mask, mask_sha = (None, None) if args.mask is None else _load_mask(args.mask, args.max_cells)
            padding, padding_sha = ((None, None) if args.key_padding is None else
                                    _load_mask(args.key_padding, args.max_cells, 'key-padding mask'))
            result = audit(mask, policy, max_cells=args.max_cells, key_padding_mask=padding,
                           **options, **kwargs).to_dict()
            result['source'] = {'kind': 'explicit local inputs', 'contract_sha256': contract_sha,
                                'mask_sha256': mask_sha, 'padding_sha256': padding_sha}
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        reports = []
        if args.json:
            reports.append((args.json, text + '\n'))
        if args.html:
            reports.append((args.html, render_report(result)))
        publish_reports(reports, inputs=inputs, force=args.force)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0 if result['status'] == 'PASS' else 1
    except (ContractError, OSError, ValueError, TypeError, UnicodeError) as exc:
        print(json.dumps({'schema_version': 1, 'status': 'ERROR', 'error': str(exc)}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
