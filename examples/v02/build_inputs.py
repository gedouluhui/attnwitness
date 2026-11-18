"""Create the recorded v0.2 example inputs; refuses to replace existing files."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def write_json(path, payload):
    if path.exists():
        raise SystemExit(f'{path.name} already exists; refusing to replace it')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(path.relative_to(HERE))


def save(path, array):
    if path.exists():
        raise SystemExit(f'{path.name} already exists; refusing to replace it')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    print(path.relative_to(HERE))


def prefix():
    write_json(HERE / 'prefix' / 'spec.json', {
        'mode': 'prefix',
        'tokens': [{'id': 'prefix-0', 'segment': 'document', 'position': 0, 'prefix': True},
                   {'id': 'prefix-1', 'segment': 'document', 'position': 1, 'prefix': True},
                   {'id': 'completion', 'segment': 'document', 'position': 2}]})
    save(HERE / 'prefix' / 'mask.npy',
         np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool))


def combined_overflow():
    write_json(HERE / 'combined-overflow' / 'spec.json', {
        'mode': 'bidirectional', 'backend': 'mha', 'batch_size': 1, 'heads': 1,
        'tokens': [{'id': 'x', 'segment': 'document', 'position': 0},
                   {'id': 'y', 'segment': 'document', 'position': 1}]})
    save(HERE / 'combined-overflow' / 'attention.npy',
         np.array([[0, -40000], [0, 0]], dtype=np.float16))
    save(HERE / 'combined-overflow' / 'padding.npy',
         np.array([[0, -40000]], dtype=np.float16))


def late_witness():
    n = 24
    batches = [[{'id': f'b{b}t{i}', 'segment': 'document', 'position': i} for i in range(n)]
               for b in range(2)]
    write_json(HERE / 'late-witness' / 'spec.json', {
        'mode': 'causal', 'backend': 'sdpa', 'batch_size': 2, 'heads': 2, 'tokens': batches})
    mask = np.broadcast_to(np.tril(np.ones((n, n), dtype=bool)), (2, 2, n, n)).copy()
    mask[1, 1, 20, 21] = True
    save(HERE / 'late-witness' / 'mask.npy', mask)


if __name__ == '__main__':
    prefix()
    combined_overflow()
    late_witness()
