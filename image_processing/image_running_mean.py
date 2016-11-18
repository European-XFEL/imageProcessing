#!/usr/bin/env python

import numpy
from collections import deque


class ImageRunningMean:
    def __init__(self):
        self.__runningMean = None  # Image running mean
        self.__imageQueue = deque([])  # Image queue

    def append(self, image, maxlen=None):
        '''Append an image to the queue, update running mean'''

        if not isinstance(image, numpy.ndarray):
            raise ValueError("image has incorrect type: %s" % str(type(image)))

        if maxlen is not None and maxlen < 1:
            raise ValueError("maxlen must be positive but is equal %s" %
                             maxlen)

        _n = len(self.__imageQueue)  # current length
        imageCopy = numpy.copy(image.astype('float64'))  # Copy the image

        if _n == 0:
            # Mean coincides with current image
            self.__imageQueue.append(imageCopy)
            self.__runningMean = imageCopy.copy()
            return

        if imageCopy.shape != self.shape:
            raise ValueError("image has incorrect shape: %s != %s" %
                             (str(imageCopy.shape), str(self.shape)))

        _sum = _n * self.__runningMean
        if maxlen is not None and _n >= maxlen:
            # Images must be dropped

            if _n < 2 * (maxlen - 1):
                # Pop images from queue and update running total
                while len(self.__imageQueue) >= maxlen:
                    _sum -= self.__imageQueue.popleft()
            else:
                # More convenient to re-calculate sum
                while len(self.__imageQueue) >= maxlen:
                    self.__imageQueue.popleft()
                _sum = sum(self.__imageQueue)

        # Append new image and update running mean
        self.__imageQueue.append(imageCopy)
        self.__runningMean = (_sum + imageCopy) / len(self.__imageQueue)

    def popleft(self):
        '''Pop an image from the queue, update the running mean'''

        _n = len(self.__imageQueue)
        if _n == 0:
            return
        elif _n == 1:
            # No images will be left
            self.__imageQueue.clear()
            self.__runningMean = None
        elif _n == 2:
            # Only one image will be left
            # Mean coincides with last image left
            self.__imageQueue.popleft()
            self.__runningMean = numpy.copy(self.__imageQueue[0])
        else:
            image = self.__imageQueue.popleft()
            self.__runningMean = (_n * self.__runningMean - image) / (_n - 1)

    def clear(self):
        '''Clear the queue and reset the running mean'''
        self.__imageQueue.clear()
        self.__runningMean = None

    def recalculate(self):
        '''Recalculate the mean'''

        _n = len(self.__imageQueue)
        if _n > 0:
            self.__runningMean = sum(self.__imageQueue) / _n

    @property
    def runningMean(self):
        '''Return the running Mean'''
        return self.__runningMean

    @property
    def size(self):
        '''Return the size of the queue'''
        return len(self.__imageQueue)

    @property
    def shape(self):
        '''Return the shape of images in the queue'''
        if self.size == 0:
            return ()
        else:
            return self.__imageQueue[0].shape
