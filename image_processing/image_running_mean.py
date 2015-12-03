#!/usr/bin/env python

__author__="andrea.parenti@xfel.eu"
__date__ ="December  3, 2015, 10:49 AM"
__copyright__="Copyright (c) 2010-2015 European XFEL GmbH Hamburg. All rights reserved."

import numpy
from collections import deque

class ImageRunningMean:

    def __init__(self):
        self.__imageQueue = deque([])  # Image queue
        self.__runningMean = None  # Image running mean

    def append(self, image):
        '''Append an image to the queue, update running mean'''

        if not isinstance(image, numpy.ndarray):
            raise ValueError("image has incorrect type: %s" % str(type(image)))

        _n = len(self.__imageQueue)
        imageCopy = numpy.copy(image.astype('float64'))  # Make a copy of the image
        if _n==0:
            self.__imageQueue.append(imageCopy)
            self.__runningMean = imageCopy  # Mean coincides with current image
        else:
            if imageCopy.shape==self.shape():
                self.__runningMean = ( _n* self.__runningMean + imageCopy) / (_n+1)
                self.__imageQueue.append(imageCopy)
            else:
                raise ValueError("image has incorrect shape: %s != %s" % (str(imageCopy.shape), str(self.shape())))

    def popleft(self):
        '''Pop an image from the queue, update the running mean'''

        _n = len(self.__imageQueue)
        if _n==0:
            return
        elif _n==1:
            # No images will be left
            self.__imageQueue.clear()
            self.__runningMean = None
        elif _n==2:
            # Only one image will be left
            self.__imageQueue.popleft()
            self.__runningMean = numpy.copy(self.__imageQueue[0])  # Mean coincides with last image left
        else:
            image = self.__imageQueue.popleft()
            self.__runningMean = ( _n*self.__runningMean - image) / (_n-1)

    def clear(self):
        '''Clear the queue and reset the running mean'''
        self.__imageQueue.clear()
        self.__runningMean = None

    def recalculate(self):
        '''Recalculate the mean'''

        _n = len(self.__imageQueue)
        if _n>0:
            self.__runningMean = sum(self.__imageQueue) / _n

    def runningMean(self):
        '''Return the running Mean'''
        return self.__runningMean

    def size(self):
        '''Return the size of the queue'''
        return len(self.__imageQueue)

    def shape(self):
        '''Return the shape of images in the queue'''
        if self.size()==0:
            return ()
        else:
            return self.__imageQueue[0].shape
