<!-- omit in toc -->
# Change Detection

This directory contains the code for change detection.

<!-- omit in toc -->
## Table of Contents

<!-- markdownlint-disable line-length -->
- [1. Supported Models](#1-supported-models)
- [2. Usage](#2-usage)
  - [2.1. Dataset Preparation](#21-dataset-preparation)
  - [2.2. Loader](#22-loader)
  - [2.3. Running Change Detection](#23-running-change-detection)
    - [2.3.1. Change Detection on a Pair of Images with `detect()`](#231-change-detection-on-a-pair-of-images-with-detect)
    - [2.3.2. Change Detection on a Dataset with `run_all()`](#232-change-detection-on-a-dataset-with-run_all)
<!-- markdownlint-enable line-length -->


## 1. Supported Models

Change detection does not use any machine learning models but depends on the
results of object detection. Therefore, part of the change detection pipeline
should be modified depending on the object detection model used.

Currently, the following models are supported:

- [YOLO](https://docs.ultralytics.com/)


## 2. Usage


### 2.1. Dataset Preparation

For any model, prepare your dataset in the following format:

```console
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

In change detection, each direct subdirectory of `datasets/` are used as
**dataset**, and there should be exactly two dataset directories, one for
**before** the change and the other for **after** the change.

The names of the dataset directories and their parent directory can be defined
arbitrarily.

> [!WARNING]
> If there are more than or less than two dataset directories, the change
> detection process raises an error.

Each directory under a dataset directory should contain results of object
detection as well as images. Therefore, the structure depends on the object
detection model used. For example, if YOLO is used, the structure of a dataset
should be as follows:

```console
dataset/
├── image0/
│   ├── labels/
│   │   └── image0.txt
│   └── image0.jpg
├── image1/
│   └── ...
└── ...
```


### 2.2. Loader

Since the structure of the dataset depends on the object detection model used,
the change detection pipeline requires a data loader to load object detection
results and images.

A data loader should inherit from the `ObjectDetectionDataLoaderBase` class and
implement the following methods:

- `get_images_path()`
- `get_labels_path()`
- `read_labels()`

For more details, please refer to the definition of
`ObjectDetectionDataLoaderBase` in [`base.py`](../loader/base.py).


### 2.3. Running Change Detection

You can run change detection on a single pair of images or an entire dataset.


#### 2.3.1. Change Detection on a Pair of Images with `detect()`

Each change detector instance has a `run()` method to perform change detection
on a pair of images.

The following snippet shows how to run change detection on a pair of images:

```python
detector = ChangeDetector()
loader = YoloObjectDetectionDataLoader() # This varies depending on the model
detector.run(before_image_path, after_image_path, loader)
```

The `run()` method returns a `SinglePairResult` object, which contains the
results of change detection.


#### 2.3.2. Change Detection on a Dataset with `run_all()`

Each change detector instance has a `run_all()` method to perform change
detection on an entire dataset.

The following snippet shows how to run change detection on a dataset:

```python
detector = ChangeDetector()
loader = YoloObjectDetectionDataLoader() # This varies depending on the model
detector.run_all(datasets_path, loader, before_name)
```

Note that the `before_name` argument should be the name of the directory
containing images before the change. By default, it is set to `before`. The name
of the dataset after the change is automatically determined as the other
subdirectory name in `datasets_path`.

This will save the results in the `datasets_path` directory as follows:

```console
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
