# The Karabo's `imageProcessing` package

## Introduction

The `imageProcessing` package provides two Python submodules:
`image_processing.image_processing` and `image_processing.image_running_mean`.

The `image_processing.image_processing` module contains functions for basic
processing of `numpy.ndarray` images.

The `image_processing.image_running_mean` module provides the
`ImageRunningMean` class, which calculates a simple moving average or a
cumulative moving average of a `numpy.ndarray`. See the [moving average
definition](https://en.wikipedia.org/wiki/Moving_average) for details.

See the documentation for the [`image_processing`](image_processing.md) and
[`image_running_mean`](image_running_mean.md) submodules.
