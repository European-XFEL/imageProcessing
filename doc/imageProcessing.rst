.. _imageProcessing:

************************************
The Karabo's imageProcessing package
************************************

Introduction
============

The imageProcessing package provides two python submodules,
image_processing.image_processing and
image_processing.image_running_mean.

In image_processing.image_processing a number of functions is
provided, which can be used to do some basic processing on a
numpy.ndarray.

In image_processing.image_running_mean the ImageRunningMean class is
provided, which can be used to calculate a simple moving average or
a cumulative moving average (please find `here
<https://en.wikipedia.org/wiki/Moving_average>`_ the definition) of a
numpy.ndarray.


The image_processing submodule
==============================

The functions in image_processing submodule can be imported with::

    from image_processing.image_processing import *

Many of the functions will by default modify the input image: to
avoid that use the copy=False option.

The following functions are available:

.. function:: imagePixelValueFrequencies(image)

    Call internally numpy.bincount, and return the one dimensional
    ndarray of pixel value frequencies. Input image must have integer
    dtype.

.. function:: imageSetThreshold(image, threshold, copy=False)

     Pixels below threshold are set to zero.

.. function:: imageSubtractBackground(image, background, copy=False)

    Background is subtracted from image. Beware the image data type:
    It should be signed since subtraction can result in negative
    values!

.. function:: imageApplyMask(image, mask, copy=False)

    mask is applied to the image. Mask values below 0 masks out
    pixels.

.. function:: imageSelectRegion(image, x1, x2, y1, y2, copy=False)

    Everything outside the rectangular region is set to zero.

.. function:: imageSumAlongY(image)

    Image is summed along the Y axis.

.. function:: imageSumAlongX(image)

    Image is summed along the X axis.

.. function:: imageCentreOfMass(image)

    Return centre-of-mass and width of an image (1-D or 2-D). If the
    image is 1-D it returns (x0, sx); if the image is 2-D it returns
    (x0, y0, sx, sy). For the calculations numpy.average and
    numpy.sqrt are used.

.. function:: fitGauss(image, p0=None, enablePolynomial=False)

    Return the gaussian fit parameters of an image (1-D or
    2-D). Additionally a first order polynomial a*x + b*y + c can be
    added to the gaussian. Returned values are (A, x0, sx, covariance,
    error) or (A, x0, sx, a, c, covariance, error) in case of 1-D, (A,
    x0, y0, sx, sy, covariance, error) or (A, x0, y0, sx, sy, a, b, c,
    covariance, error) in case of 2-D.  The fit is done with the help
    of the scipy.optimize.leastsq function, with full_output option.

.. function:: fitGauss2DRot(image, p0=None, enablePolynomial=False)

    Return the gaussian fit parameters of a 2-D image, including
    rotation. Additionally a first order polynomial a*x + b*y + c can
    be added to the gaussian. Returned values are (A, x0, y0, sx, sy,
    theta, covariance, error) or (A, x0, y0, sx, sy, theta, a, b, c,
    covariance, error). The fit is done with the help of the
    scipy.optimize.leastsq function, with full_output option.


The ImageRunningMean class
==========================

This image_processing.image_running_mean submodule provide the
ImageRunningMean class::

    from image_processing.image_running_mean import ImageRunningMean

The available methods for the class are:

.. function:: append(image, maxlen=None)

    Append an image to the queue and update the running mean. By
    default the cumulative moving average is calculated, if you want
    the simple moving average you have to use the parameter maxlen to
    pop the excess images. The method will raise a ValueError if the
    image has not the same shape as the other ones in the queue.

.. function:: popleft()

    Pop an image from the queue, and update the running mean.

.. function:: clear()

    Clear the queue and reset the running mean.

.. function:: recalculate()

    Recalculate the mean.

.. function:: runningMean()

    Return the running mean.

.. function:: size()

    Return the size of the queue.

.. function:: shape()

    Return the shape of images in the queue.
