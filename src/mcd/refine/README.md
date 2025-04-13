<!-- omit in toc -->
# Refinement

This directory contains the code for the refinement of the change detection
results. The refinement step is optional but can remove false positives to a
great extent.

<!-- omit in toc -->
## Table of Contents

- [1. Dataset Preparation](#1-dataset-preparation)
- [2. Loader](#2-loader)
- [3. Running the Refinement Algorithm](#3-running-the-refinement-algorithm)


## 1. Dataset Preparation

The refinement step requires the original dataset, which is the dataset used for
the object detection step, with images and depth maps.

> [!NOTE]
> This is not the dataset containing the object detection results.

The dataset should be organized as follows:

```console
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

File names and extensions can vary and the directory structure does not have to
be exactly the same as above as long as the datasets directory contains exactly
two directories. However, each image should have the following data:

- RGB image
- Depth map
- Camera parameters (i.e., intrinsics and extrinsic)
- (Optional) Confidence map of depth


## 2. Loader

Similar to the change detection step, the refinement step requires a data loader
to load the data from the dataset.

A data loader should inherit from the `RefinementDataLoaderBase` class and
implement the following methods:

- `get_depth_map_path_pair()`
- `get_depth_map()`
- `load_camera_parameters()`

For more details, please refer to the definition of
`RefinementDataLoaderBase` in [`base.py`](../loader/base.py).


## 3. Running the Refinement Algorithm

Refinement is a two-step process:

1. Convert change detection results to 3D representation.
2. Refine the 3D representation by clustering.

The following snippet shows how to run the refinement algorithm:

```python
refiner = Refiner()

# 1. Convert change detection results to 3D representation
results_3d = refiner.convert_result_2d_to_3d(
    datasets_path,
    results,
    loader=MyRefinementDataLoader(),
    before_name=before_name,
)
# 2. Refine the 3D representation by clustering
refiner.refine(results_3d, results_path)
```

This will save the refined results in the `results_path` directory.

```console
results_path/
├── refined_change_detection_result_3d.json
└── refined_change_points.svg
```

`refined_change_detection_result_3d.json` contains the refined change detection
results in 3D space, and `refined_change_points.svg` is a visualization of
the refined change points.

Also, the change detection results in 3D space and its visualization before
refinement are saved in the datasets directory as follows:

```console
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
├── before/
│   ├── images/
│   │   ├── image_0.png
│   │   └── ...
│   ├── cameras/
│   │   ├── image_0.json
│   │   └── ...
│   └── depth/
│       ├── image_0.exr
│       └── ...
├── change_detection_result_3d.json
└── change_points.svg
```
