"""Object detection package.

Dataset
-------

For any object detection model, the dataset directory is expected to have the following
structure:

```
dataset_id/
    └── images/
        ├── image0.png
        ├── image1.png
        └── ...
```

where image names can vary depending on the dataset, but it is important that the images
are in the `images` directory.


Object Detector
---------------

The `ObjectDetectorBase` class should be inherited to implement specific object
detection models. Ensure that your implementation adheres to the expected interface and
methods.

```python
from mcd.object_detection import base


class MyObjectDetector(base.ObjectDetectorBase):
    # Implement the required methods
    ...
```

Run Object Detection
--------------------

To run the object detection, you can use the following code:

```python
import pathlib

model_path = pathlib.Path("path/to/model")
detector = MyObjectDetector(model_path)
dataset_path = pathlib.Path("path/to/dataset")
results_path = pathlib.Path("path/to/results")
detector.detect_all(dataset_path, results_path)
```

This will run the object detection on all images in the dataset and save the results in
the specified directory.

```
results_path/
└── dataset_id/
    ├── image0/
    │   ├── labels/
    │   │   └── image0.txt
    │   └── image0.jpg
    ├── image1/
    │   ├── labels/
    │   │   └── image1.txt
    │   └── image1.jpg
    └── ...
```

"""
