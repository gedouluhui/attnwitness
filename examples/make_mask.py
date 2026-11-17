"""Generate an original, two-token NPY example next to this script."""
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    path = Path(__file__).with_name('causal.npy')
    if path.exists():
        raise SystemExit('causal.npy already exists; refusing to replace it')
    np.save(path, np.array([[True, False], [True, True]]))
    print(path.name)
