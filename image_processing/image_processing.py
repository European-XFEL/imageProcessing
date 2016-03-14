#!/usr/bin/env python

__author__="andrea.parenti@xfel.eu"
__date__ ="October 29, 2013, 15:33 AM"
__copyright__="Copyright (c) 2010-2013 European XFEL GmbH Hamburg. All rights reserved."

import math
import numpy
import scipy
import scipy.optimize
import scipy.stats


def imagePixelValueFrequencies(image):
    """Returns the distribution of the pixel value freqiencies"""
    if not isinstance(image, numpy.ndarray):
        return None
    
    return numpy.bincount(image.reshape(image.size))


def imageSetThreshold(image, threshold, copy=False):
    """Sets to 0 image elements below threshold"""
    if not isinstance(image, numpy.ndarray):
        return None

    if copy:
        _image = image.copy()
    else:
        _image = image

    # return scipy.stats.threshold(_image, threshold)

    # #  A bit faster than scipy.stats.threshold
    # _image[_image<threshold] = 0
    # return _image

    # A bit faster than m[m<threshold] = 0
    _image *= _image>threshold
    return _image


def imageSubtractBackground(image, background, copy=False):
    """Subtract background from image. Beware the image data type: It should 
    be signed since subtraction can result in negative values!"""
    if not isinstance(image, numpy.ndarray):
        return None

    if not isinstance(background, numpy.ndarray):
        return None

    if background.shape!=image.shape:
        return None

    if copy:
        _image = image.copy()
    else:
        _image = image

    _image -= background  # subtract bkg from image

    return _image


def imageApplyMask(image, mask, copy=False):
    """Apply mask to an image"""
    if not isinstance(image, numpy.ndarray):
        return None

    if not isinstance(mask, numpy.ndarray):
        return None

    if mask.shape!=image.shape:
        return None

    if copy:
        _image = image.copy()
    else:
        _image = image

    # Apply mask
    n = (mask<=0)  # Mask equal or below zero
    _image[n] = 0 # zero img, where mask is <= 0

    return _image


def imageSelectRegion(image, x1, x2, y1, y2, copy=False):
    """Select rectangular region from an image"""

    if not isinstance(image, numpy.ndarray) or \
            (image.ndim!=2 and image.ndim!=3):
        return None

    if copy:
        _image = image.copy()
    else:
        _image = image

    if _image.ndim==2:
        _image[:y1,:] = 0
        _image[y2:,:] = 0
        _image[:,:x1] = 0
        _image[:,x2:] = 0

    elif _img.ndim==3:
        _image[:y1,:,:] = 0
        _image[y2:,:,:] = 0
        _image[:,:x1,:] = 0
        _image[:,x2:,:] = 0

    return _image


def imageSumAlongY(image):
    """Sums image along Y axis"""
    if not isinstance(image, numpy.ndarray) or image.ndim!=2:
        return None
    
    return image.sum(axis=0)


def imageSumAlongX(image):
    """Sums image along X axis"""
    if not isinstance(image, numpy.ndarray) or image.ndim!=2:
        return None
    
    return image.sum(axis=1)


def imageCentreOfMass(image):
    """Returns centre-of-mass and widths of an image (1-D or 2-D)"""
    if not isinstance(image, numpy.ndarray):
        return None
    
    if image.ndim==1:
        # 1-D image
        
        values = numpy.arange(image.size)
        weights = image
        
        # Centre-of-mass and width
        x0 = numpy.average(values, weights=weights)
        sx = numpy.average((values-x0)**2, weights=weights)
        sx = numpy.sqrt(sx)
        
        return (x0, sx)
    
    elif image.ndim==2:
        # 2-D image

        # sum over y, evaluate centre-of-mass and width
        imgX = image.sum(axis=0)
        (x0, sx) = imageCentreOfMass(imgX)
        
        # sum over x, evaluate centre-of-mass and width
        imgY = image.sum(axis=1)
        (y0, sy) = imageCentreOfMass(imgY)
        
        return (x0, y0, sx, sy)
    
    else:
        return None


def fitGauss(image, p0=None, enablePolynomial=False):
    """Returns gaussian fit parameters of an image (1-D or 2-D).
    Additionally add 1st order polynomial a*x + b*x +c."""
    if not isinstance(image, numpy.ndarray):
        return None

    if image.ndim==1:
        # 1-D image

        if p0 is None:
            # Evaluates initial parameters
            (x0, sx) = imageCentreOfMass(image)
            if enablePolynomial is False:
                p0 = (image.max(), x0, sx)
            else:
                p0 = (image.max(), x0, sx, 0., 0.)

        # [AP] scipy.optimize.leastsq assumes equal errors
        x = numpy.arange(image.size)
        errFun = lambda p: gauss1d(x, *p, enablePolynomial=enablePolynomial) - image
        out = scipy.optimize.leastsq(errFun, p0, full_output=1)

        s_sq = (out[2]['fvec']**2).sum()/(len(out[2]['fvec'])-len(out[0])) # residual variance
        p1 = out[0] # parameters
        if out[1] is not None:
            p1cov = s_sq * out[1] # parameters covariance matrix
        else:
             p1cov = None # singular matrix encountered
        ier = out[4] # error

        return p1, p1cov, ier

    elif image.ndim==2:
        # 2-D image

        if p0==None:
            # Evaluates initial parameters
            (x0, y0, sx, sy) = imageCentreOfMass(image)
            if enablePolynomial is False:
                p0 = (image.max(), x0, y0, sx, sy)
            else:
                p0 = (image.max(), x0, y0, sx, sy, 0. ,0., 0.)

        # [AP] scipy.optimize.leastsq assumes equal errors
        x = numpy.arange(image.shape[1]).reshape(1, image.shape[1])
        y = numpy.arange(image.shape[0]).reshape(image.shape[0], 1)
        errFun = lambda p: numpy.ravel(gauss2d(x, y, *p, enablePolynomial=enablePolynomial) - image)
        out = scipy.optimize.leastsq(errFun, p0, full_output=1)

        s_sq = (out[2]['fvec']**2).sum()/(len(out[2]['fvec'])-len(out[0])) # residual variance
        p1 = out[0] # parameters
        if out[1] is not None:
            p1cov = s_sq * out[1] # parameters covariance matrix
        else:
            p1cov = None # singular matrix encountered
        ier = out[4] # error

        return p1, p1cov, ier

    else:
        return None


def fitGauss2DRot(image, p0=None, enablePolynomial=False):
    """Returns gaussian fit parameters of a 2-D image.
    Additionally add 1st order polynomial a*x + b*x +c."""
    if not isinstance(image, numpy.ndarray) or image.ndim!=2:
        return None

    if p0==None:
        # Evaluates initial parameters
        (x0, y0, sx, sy) = imageCentreOfMass(image)
        if enablePolynomial is False:
            p0 = (image.max(), x0, y0, sx, sy, 0.)
        else:
            p0 = (image.max(), x0, y0, sx, sy, 0., 0., 0., 0.)

    # [AP] scipy.optimize.leastsq assumes equal errors
    x = numpy.arange(image.shape[1]).reshape(1, image.shape[1])
    y = numpy.arange(image.shape[0]).reshape(image.shape[0], 1)
    errFun = lambda p: numpy.ravel(gauss2dRot(x, y, *p, enablePolynomial=enablePolynomial) - image)
    out = scipy.optimize.leastsq(errFun, p0, full_output=1)

    s_sq = (out[2]['fvec']**2).sum()/(len(out[2]['fvec'])-len(out[0])) # residual variance
    p1 = out[0] # parameters
    if out[1] is not None:
        p1cov = s_sq * out[1] # parameters covariance matrix
    else:
        p1cov = None # singular matrix encountered
    ier = out[4] # error

    return p1, p1cov, ier


def gauss1d(x, height, x0, sx, a=0., b=0., enablePolynomial=False):
    """Returns a gaussian + 1st order polynomial, with the given parameters.
    The gaussian is in the form height*numpy.exp(-(((x-x0)/sx)**2)/2).
    The polynomial is in the form a*x + b."""
    #
    f = height * numpy.exp(-0.5 * ((x-float(x0))/sx)**2)
    #
    if enablePolynomial is True:
        # Add polynomial
        if a!=0.:
            f += a*x  # In-place
        if b!=0:
            f += b  # In-place
    #
    return f


def gauss2d(x, y, height, x0, y0, sx, sy, a=0., b=0., c=0., enablePolynomial=False):
    """2-d gaussian + 1st order polynomial, with the given parameters.
    The gaussian is in the form height*numpy.exp(-(((x-x0)/sx)**2+((y-y0)/sy)**2)/2).
    The polynomial is in the form a*x + b*y +c."""
    #
    f = height * numpy.exp(-0.5 * (((x-float(x0))/sx)**2+((y-float(y0))/sy)**2))
    #
    if enablePolynomial is True:
        # Add polynomial
        if a!=0.:
            f += a*x  # In-place
        if b!=0:
            f += b*y  # In-place
        if c!=0:
            f += c  # In-place
    #
    return f


def gauss2dRot(x, y, height, x0, y0, sx, sy, theta_deg, a=0., b=0., c=0., enablePolynomial=False):
    """2-d rotated gaussian + 1st order polynomial, with the given parameters.
    The gaussian is in the form height*numpy.exp(-(((x'-x0)/sx)**2+((y'-y0)/sy)**2)/2).
    The polynomial is in the form a*x + b*y +c."""
    #
    theta = math.radians(theta_deg)  # rad -> deg
    # Reference frame change: First translate origin to (x0, y0)...
    x1 = x - x0
    y1 = y - y0
    # ... then rotate axes
    x2 = x1*math.cos(theta) - y1*math.sin(theta)
    y2 = x1*math.sin(theta) + y1*math.cos(theta)
    #
    f = gauss2d(x2, y2, height, 0., 0., sx, sy, a, b, c, enablePolynomial)
    #
    return f
