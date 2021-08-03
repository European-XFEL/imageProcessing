import numpy as np


class ImageExponentialRunnningAverage:
    """Simple, fast and efficient running average method, widely used in
    machine learning to track running statistics. It does not need to store
    a 100000 image ringbuffer: the running average is held by a single numpy
    array with the same size as the image and updated as the weighted average
    of the previous state and the new frame according to:
    ```
    AVG_new = w*IMG_new + (1-w)*AVG_old
    ```
    The number of averaged frames sets the decay rate and can be changed
    without clearing the buffer, i.e. you can start with a faster decay and
    slow it down after initial convergence. The weighted average is stored as
    a float64 array and must be converted back to the image type.
    """

    def __init__(self):
        self.__nimages = 1.0
        self.__mean = None

    @property
    def __tau(self):
        """The decay rate is the inverse of the number of frames."""
        return 1.0 / self.__nimages

    def clear(self):
        """Reset the mean"""
        self.__mean = None

    def append(self, image, n_images):
        """Add a new image to the average"""
        # Check for correct type and input values
        if not isinstance(image, np.ndarray):
            raise ValueError("Image has incorrect type: %s" % str(type(image)))
        if n_images <= 0:
            raise ValueError("The averager's smoothing rate must be positive "
                             "instead of %f." % n_images)

        # We assign the smoothing coefficient
        self.__nimages = n_images

        if self.__mean is None:
            # If running average is empty, we explicitly assign fp64
            self.__mean = image.astype(np.float64)
        else:
            # If it's already running, just update the state
            self.__mean = self.__tau * image + (1.0 - self.__tau) * self.__mean

    @property
    def mean(self):
        """Returns the current mean"""
        return self.__mean

    @property
    def size(self):
        """Return the inverse decay rate"""
        return self.__nimages

    @property
    def shape(self):
        if self.__mean is None:
            return ()
        else:
            return self.__mean.shape
