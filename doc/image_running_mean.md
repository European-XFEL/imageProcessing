# The `ImageRunningMean` class

The `image_processing.image_running_mean` submodule provides the
`ImageRunningMean` class, which can be imported by:

```python
from image_processing.image_running_mean import ImageRunningMean
```

`ImageRunningMean` calculates a simple moving average or a
cumulative moving average of a NumPy array. See the [moving average
definition](https://en.wikipedia.org/wiki/Moving_average) for details.

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
