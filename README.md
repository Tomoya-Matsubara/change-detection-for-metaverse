<!-- omit in toc -->
# Change Detection for Constantly Maintaining Up-to-date Metaverse Maps

This repository contains the official implementation of the paper
["Change Detection for Constantly Maintaining Up-to-date Metaverse Maps"](
    https://ieeexplore.ieee.org/document/10536152).


<div align="center">
    <!-- markdownlint-disable MD013 -->
    <!-- cSpell: ignore autoplay -->
    <video controls loop autoplay muted src="https://github.com/user-attachments/assets/a9d0f188-b197-4ca5-a297-56215e1efc93"></video>
    <!-- markdownlint-enable MD013 -->
</div>


<!-- omit in toc -->
## Abstract

*Metaverse has been attracting more and more attention because of its potential
for various use cases. In metaverse applications, the seamless integration of
digital and physical worlds is vital for synchronizing information from one
world to another. One way to achieve this is to reconstruct 3D environmental
maps every time, which is not feasible due to computational complexity.
A cheaper alternative is to detect what objects have changed and update only
the changed objects. To build the foundation of the change detection algorithm
for that, in this paper, we propose a change detection method combined with
object classification. Despite its simplicity, the experiment showed promising
results with an object detector fine-tuned with data from the target
environment. Furthermore, with our clustering-based post-processing, false
positives produced by the frame-wise change detection were observed to be
successfully suppressed.*

<!-- omit in toc -->
## Table of Contents

<!-- markdownlint-disable line-length -->
- [1. Prerequisites](#1-prerequisites)
- [2. Datasets](#2-datasets)
- [3. Setup](#3-setup)
- [4. Running the Change Detection Algorithm](#4-running-the-change-detection-algorithm)
  - [4.1. Object Detection](#41-object-detection)
- [5. Setup for Development](#5-setup-for-development)
  - [5.1. `pnpm`](#51-pnpm)
  - [5.2. `Lefthook`](#52-lefthook)
<!-- markdownlint-enable line-length -->

## 1. Prerequisites

The change detection algorithm in this repository depends on pre-trained object
detection models. Please prepare your own object detection model.

Currently, the following object detection models are supported:

- YOLO


## 2. Datasets

Throughout the documentation, the term "dataset" is used to refer to a directory
containing relevant data for each scene. Each dataset is required to have the
following data:

- RGB-D images
- Camera parameters (intrinsic and extrinsic parameters)

Optionally, the dataset can contain the following data:

- Correction matrix

The correction matrix in this context refers to the matrix that corrects the
the position in camera space while calculating the 3D position of pixels in
images.

> [!TIP]
> The correction matrix is only required in limited cases. For example,
> ARKit uses the portrait-oriented coordinate system, which requires a
> correction matrix to convert it to the right-handed coordinate system.

In most cases, it is not necessary to provide the correction matrix
(i.e., the identity matrix can be used).


## 3. Setup

The project is fully containerized. To set up the project, run the following
command:

```bash
docker compose watch
```

<!-- markdownlint-disable line-length -->
This command will start the container with dependencies installed. To work in
the container, the Visual Studio Code
[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension is recommended.
<!-- markdownlint-enable line-length -->

## 4. Running the Change Detection Algorithm

The change detection pipeline is composed of the following steps:

1. [Object Detection](#41-object-detection)


### 4.1. Object Detection

The first step is object detection, detecting objects in the input images both
before and after the change. In this step, a pre-trained object detection model
is required.

Depending on your model, you need to implement an object detector class that
inherits from the `ObjectDetectorBase` class in
[object_detection/base.py](./src/mcd/object_detection/base.py). For more details
on how to implement an object detector, refer to the
[`README.md`](./src/mcd/object_detection/README.md) file in the
`object_detection` directory.

Once you have implemented the object detector, you can run the object detection
step as follows:

```python
detector = YoloObjectDetector(model_path) # This varies depending on the model
detector.detect_all(dataset_path, results_path)
```


## 5. Setup for Development


### 5.1. `pnpm`

While this project is a Python project, this repository uses JavaScript packages
for code formatting and linting. As a package manager, `pnpm` is used.

You can install `pnpm` by following
[the installation guide](https://pnpm.io/installation).

After installing `pnpm`, run the following command to install the JavaScript
dependencies:

```bash
pnpm install --frozen-lockfile
```


### 5.2. `Lefthook`

This repository uses `Lefthook` as a Git hooks manager. You can install
`Lefthook` by following
[the installation guide](https://lefthook.dev/installation/).

After installing `Lefthook`, run the following command to set up the Git hooks:

```bash
lefthook install
```
