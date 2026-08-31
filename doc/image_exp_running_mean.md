# The `ImageExponentialRunnningAverage` class

The `image_processing.image_exp_running_average` submodule provides the
`ImageExponentialRunnningAverage` class, which can be imported with:

```python
from image_processing.image_exp_running_average import (
    ImageExponentialRunnningAverage)
```

`ImageExponentialRunnningAverage` calculates an exponential running average
using a single NumPy `float64` array. For each appended image, the average is
updated according to:

```text
mean_new = weight * image + (1 - weight) * mean_old
```

where `weight` is the inverse of `n_images`. All appended images must be NumPy
arrays with the same shape.

See [here](https://en.wikipedia.org/wiki/Exponential_smoothing) for the
definition and more details.

## Available methods and properties

#### `append(image, n_images)`
Appends an image and updates the exponential running average. `n_images` must
be positive and controls the smoothing rate: larger values produce a smaller
weight for the new image. Raises `ValueError` for an invalid image type or
non-positive smoothing rate.

#### `clear()`
Clears the accumulated mean. The smoothing rate is retained.

#### `mean`
Returns the current mean, or `None` when no image has been appended.

#### `size`
Returns the current smoothing parameter (`n_images`).

#### `shape`
Returns the shape of the images in the mean, or `()` when empty.
