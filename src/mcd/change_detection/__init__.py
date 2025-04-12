"""Change detection package.

Dataset
-------

The dataset directory is expected to have the following structure:

```
datasets/
├── after/
│   ├── image0/
│   │   └── ...
│   ├── image1/
│   │   └── ...
│   └── ...
└── before/
    ├── image0/
    │   └── ...
    ├── image1/
    │   └── ...
    └── ...
```

where the contents under each image directory vary depending on the model. The names
of the dataset directories and their parent directory can be defined arbitrarily,
but each image directory should contain object detection results for the corresponding
image as well as the image file itself.


Data Loader
-----------

Since how to load the object detection results is different for each model, the
`ObjectDetectionDataLoaderBase` class should be inherited to implement specific data
loaders. Ensure that your implementation adheres to the expected interface and methods.

```python
from mcd.loader import base


class MyObjectDetectionDataLoader(base.ObjectDetectionDataLoaderBase):
    # Implement the required methods
    ...
```

Run Change Detection
--------------------

To run the change detection, you can use the following code:

```python
import pathlib

detector = ChangeDetector()
loader = MyObjectDetectionDataLoader()
datasets_path = pathlib.Path("path/to/datasets")
before_name = "before"
detector.run_all(datasets_path, loader, before_name)
```

This will run the change detection on all pairs of images in the datasets directory and
save the results in a JSON file in the same directory.

```
datasets/
├── after/
│   ├── image0/
│   │   └── ...
│   ├── image1/
│   │   └── ...
│   └── ...
├── before/
│   ├── image0/
│   │   └── ...
│   ├── image1/
│   │   └── ...
│   └── ...
└── change_detection_result.json
```

"""
