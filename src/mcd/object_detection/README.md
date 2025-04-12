<!-- omit in toc -->
# Object Detection

This directory contains the code for object detection.

<!-- omit in toc -->
## Table of Contents

<!-- markdownlint-disable line-length -->
- [1. Supported Models](#1-supported-models)
- [2. Usage](#2-usage)
  - [2.1. Dataset Preparation](#21-dataset-preparation)
  - [2.2. Inference](#22-inference)
    - [2.2.1. Inference on an Image with `detect()`](#221-inference-on-an-image-with-detect)
    - [2.2.2. Inference on a Dataset with `detect_all()`](#222-inference-on-a-dataset-with-detect_all)
- [3. Add a New Model](#3-add-a-new-model)
<!-- markdownlint-enable line-length -->


## 1. Supported Models

Currently, the following models are supported:

- [YOLO](https://docs.ultralytics.com/)


## 2. Usage


### 2.1. Dataset Preparation

For any model, prepare your dataset in the following format:

```console
dataset_id/
    └── images/
        ├── image0.png
        ├── image1.png
        └── ...
```

You can name images in any way you like, but it is important to have them in the
`images` directory.

> [!IMPORTANT]
> The name of the `images` directory is fixed. Do not change it.


### 2.2. Inference

You can run inference on an image or an entire dataset.


#### 2.2.1. Inference on an Image with `detect()`

Each object detector instance has a `detect()` method to perform object
detection on a single image.

The following snippet shows how to run inference on an image:

```python
detector = YoloObjectDetector(model_path) # This varies depending on the model
image = cv2.imread(image_path)
detector.detect(
    image, dataset_id="my_dataset", image_id="my_image", results_path=results_path
)
```

This will save the results in the `results_path` directory as follows:

```console
results_path/
└── my_dataset/
    └── my_image/
        ├── labels/
        │   └── my_image.txt
        └── my_image.jpg
```

Note that contents under `my_image` may vary depending on the model.


#### 2.2.2. Inference on a Dataset with `detect_all()`

Each object detector instance has a `detect_all()` method to perform object
detection on an entire dataset.

The following snippet shows how to run inference on a dataset:

```python
detector = YoloObjectDetector(model_path) # This varies depending on the model
detector.detect_all(dataset_path, results_path)
```

This will save the results in the `results_path` directory as follows:

```console
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


## 3. Add a New Model

Each object detector should inherit from `ObjectDetectorBase` and implement the
following methods:

- `detect()`
- `detect_all()`

Please refer to the definition of `ObjectDetectorBase` in [`base.py`](base.py)
for more details about what each method should do.
