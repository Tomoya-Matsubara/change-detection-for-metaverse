"""Refinement package.

Dataset
-------

The dataset directory is expected to have the following structure:

```
datasets/
├── after/
│   ├── images/
│   │   ├── image_0.png
│   │   └── ...
│   ├── cameras/
│   │   ├── image_0.json
│   │   └── ...
│   └── depth/
│       ├── image_0.exr
│       └── ...
└── before/
    ├── images/
    │   ├── image_0.png
    │   └── ...
    ├── cameras/
    │   ├── image_0.json
    │   └── ...
    └── depth/
        ├── image_0.exr
        └── ...
```

The structure or file names is flexible as long as the dataset directory contains
exactly two directories: one for the dataset before the change and one for the
dataset after the change. Each image should have the following data:

- RGB image
- Depth map
- Camera parameters (i.e., intrinsic and extrinsic)
- (Optional) Confidence map of depth


Data Loader
-----------

To load data from arbitrary structures, the `RefinementDataLoaderBase` class
should be inherited to implement specific data loaders. Ensure that your
implementation adheres to the expected interface and methods.

```python
from mcd.loader import base


class MyDataLoader(base.RefinementDataLoaderBase):
    # Implement the required methods
    ...
```

Run Refinement
--------------

To run the refinement, you can use the following code:

```python
import json
import pathlib

from mcd.change_detection import result

change_detection_results_path = pathlib.Path("path/to/change_detection_results")
with change_detection_results_path.open("r") as file:
    results = result.ChangeDetectionResults.model_validate(json.load(file))

refiner = Refiner()
datasets_path = pathlib.Path("path/to/datasets")
loader = MyDataLoader()
results_path = pathlib.Path("path/to/results")
results_3d = refiner.convert_result_2d_to_3d(
    datasets_path, results, loader, before_name="before"
)
refiner.refine(results_3d, results_path)
```

This will save the refined results in the `results_path` directory.

```
results_path/
├── refined_change_detection_result_3d.json
└── refined_change_points.svg
```

`refined_change_detection_result_3d.json` contains the refined change detection
results in 3D space, and `refined_change_points.svg` is a visualization of
the refined change points.

"""
