<!-- omit in toc -->
# Change Detection for Constantly Maintaining Up-to-date Metaverse Maps

This repository contains the official implementation of the paper
["Change Detection for Constantly Maintaining Up-to-date Metaverse Maps"](
    https://ieeexplore.ieee.org/document/10536152).

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

- [1. Setup for Development](#1-setup-for-development)
  - [1.1. `pnpm`](#11-pnpm)
  - [1.2. `Lefthook`](#12-lefthook)


## 1. Setup for Development


### 1.1. `pnpm`

While this project is a Python project, this repository uses JavaScript packages
for code formatting and linting. As a package manager, `pnpm` is used.

You can install `pnpm` by following
[the installation guide](https://pnpm.io/installation).

After installing `pnpm`, run the following command to install the JavaScript
dependencies:

```bash
pnpm install --frozen-lockfile
```


### 1.2. `Lefthook`

This repository uses `Lefthook` as a Git hooks manager. You can install
`Lefthook` by following
[the installation guide](https://lefthook.dev/installation/).

After installing `Lefthook`, run the following command to set up the Git hooks:

```bash
lefthook install
```
