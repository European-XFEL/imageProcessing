# The `ImageRunningMean` class

The `image_processing.image_running_mean` submodule provides:

```python
from image_processing.image_running_mean import ImageRunningMean
```

## Available methods

#### `append(image, maxlen=None)`
Appends an image and updates the running mean. Without `maxlen` this is a
cumulative moving average; with `maxlen` it is a simple moving average.

#### `popleft()`
Pops an image and updates the running mean.

#### `clear()`
Clears the queue and resets the running mean.

#### `recalculate()`
Recalculates the mean.

#### `runningMean()`
Returns the running mean.

#### `size()`
Returns the queue size.

#### `shape()`
Returns the shape of images in the queue.
