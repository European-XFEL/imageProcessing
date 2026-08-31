# The Karabo's `imageProcessing` package

## Introduction

The `imageProcessing` package provides image-processing functions and several
image averaging classes.

The `image_processing.image_processing` module provides functions for basic
processing of NumPy arrays. It is documented [`here`](image_processing.md).

The image averaging classes are the following

- [`ImageStandardMean`](image_standard_mean.md): calculates a cumulative
  arithmetic mean;
- [`ImageExponentialRunnningAverage`](image_exp_running_mean.md): calculates
  an exponential running average;
- [`ImageRunningMean`](image_running_mean.md): calculates a simple moving
  average.


