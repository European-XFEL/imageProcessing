import numpy as np


class ImageStandardMean:
    def __init__(self):
        self.__mean = None  # Image mean
        self.__images = 0  # number of images

    def append(self, image):
        """Add a new image to the average"""
        if not isinstance(image, np.ndarray):
            raise ValueError("image has incorrect type: %s" % str(type(image)))

        # Update mean
        if self.__images > 0:
            if image.shape != self.shape:
                raise ValueError("image has incorrect shape: %s != %s" %
                                 (str(image.shape), str(self.shape)))

            self.__mean = (self.__mean * self.__images
                           + image) / (self.__images + 1)
            self.__images += 1
        else:
            self.__mean = image.astype(np.float64)
            self.__images = 1

    def clear(self):
        """Reset the mean"""
        self.__mean = None
        self.__images = 0

    @property
    def mean(self):
        """Return the mean"""
        return self.__mean

    @property
    def size(self):
        """Return the number of images in the average"""
        return self.__images

    @property
    def shape(self):
        """Return the shape of images in the average"""
        if self.size == 0:
            return ()
        else:
            return self.__mean.shape
