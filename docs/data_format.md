# Data Format

## Expected sample file

Each sample is an `.npz` file with:

- `image`: `float32` array in `[C, H, W]`
- `meta`: optional `float32` array in `[M]`

Minimal example:

```python
import numpy as np

image = np.random.randn(5, 224, 224).astype("float32")
meta = np.array([5.6, 1.0, 0.2], dtype="float32")
np.savez("sample_001.npz", image=image, meta=meta)
```

## Expected manifest columns

- `sample_path`: absolute path or path relative to the manifest file location
- `split`: usually `train`, `val`, or `test`
- `subject_name`: free-text sample id
- `data_source`: optional source tag
- one label column per task

Blank labels are allowed. Missing labels are masked automatically during multitask training.
