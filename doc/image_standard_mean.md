# The `ImageStandardMean` class

The `image_processing.image_standard_mean` submodule provides:

```python
from image_processing.image_standard_mean import ImageStandardMean
```

`ImageStandardMean` calculates the cumulative arithmetic mean of appended
images. The mean is stored as a NumPy `float64` array. All appended images
must be NumPy arrays with the same shape.

## Available methods and properties

#### `append(image)`
Appends an image and updates the cumulative mean. Raises `ValueError` when
`image` is not a NumPy array or its shape differs from the existing images.

#### `clear()`
Clears the accumulated mean and resets the image count.

#### `mean`
Returns the current mean, or `None` when no images have been appended.

#### `size`
Returns the number of images included in the mean.

#### `shape`
Returns the shape of the images in the mean, or `()` when empty.
