#!/usr/bin/env python

__author__="andrea.parenti@xfel.eu"
__date__ ="October 29, 2013, 15:33 AM"
__copyright__="Copyright (c) 2010-2013 European XFEL GmbH Hamburg. All rights reserved."

import math
import numpy
import scipy
import scipy.optimize


def imagePixelValueFrequencies(image):
    """Returns the distribution of the pixel value freqiencies"""
    if not isinstance(image, numpy.ndarray):
        return None
    
    return numpy.bincount(image.reshape(image.size))


def imageSetThreshold(image, threshold):
    """Sets a threshold on an image"""
    if not isinstance(image, numpy.ndarray):
        return None
    
    # much faster than scipy.stats.threshold()
    return image*(image>threshold)


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
        
        values = numpy.arange(image.shape[0])
        weights = image
        
        # Centre-of-mass and width
        x0 = numpy.average(values, weights=weights)
        sx = numpy.average((values-x0)**2, weights=weights)
        sx = numpy.sqrt(sx)
        
        return (x0, sx)
    
    elif image.ndim==2:
        # 2-D image

        # sum over y
        imgX = image.sum(axis=0)

        values = numpy.arange(imgX.shape[0])
        weights = imgX
        
        # Centre-of-mass and width (x)
        x0 = numpy.average(values, weights=weights)
        sx = numpy.average((values-x0)**2, weights=weights)
        sx = numpy.sqrt(sx)
        
        # sum over x
        imgY = image.sum(axis=1)
        
        values = numpy.arange(imgY.shape[0])
        weights = imgY
        
        # Centre-of-mass and width (y)
        y0 = numpy.average(values, weights=weights)
        sy = numpy.average((values-y0)**2, weights=weights)
        sy = numpy.sqrt(sy)
        
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
        errFun = lambda p: _gauss1d(x, *p, enablePolynomial=enablePolynomial) - image
        out = scipy.optimize.leastsq(errFun, p0, full_output=1)

        s_sq = (out[2]['fvec']**2).sum()/(len(out[2]['fvec'])-len(out[0])) # residual variance
        p1 = out[0] # parameters
        p1cov = s_sq * out[1] # parameters covariance matrix
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
        errFun = lambda p: numpy.ravel(_gauss2d(x, y, *p, enablePolynomial=enablePolynomial) - image)
        out = scipy.optimize.leastsq(errFun, p0, full_output=1)

        s_sq = (out[2]['fvec']**2).sum()/(len(out[2]['fvec'])-len(out[0])) # residual variance
        p1 = out[0] # parameters
        p1cov = s_sq * out[1] # parameters covariance matrix
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
    errFun = lambda p: numpy.ravel(_gauss2dRot(x, y, *p, enablePolynomial=enablePolynomial) - image)
    out = scipy.optimize.leastsq(errFun, p0, full_output=1)

    s_sq = (out[2]['fvec']**2).sum()/(len(out[2]['fvec'])-len(out[0])) # residual variance
    p1 = out[0] # parameters
    p1cov = s_sq * out[1] # parameters covariance matrix
    ier = out[4] # error

    return p1, p1cov, ier


def _gauss1d(x, height, x0, sx, a=0., b=0., enablePolynomial=False):
    """Returns a gaussian + 1st order polynomial, with the given parameters.
    The polynomial is in the form a*x + b."""
    x0 = float(x0)  # Will force f to be float, even if x is int.
    f = x - x0
    # All the following are in-place operations -> no additional ndarrays created.
    f /= sx
    f **= 2
    f /= -2
    f = numpy.exp(f)
    f *= height  # like doing f = height*numpy.exp(-(((x-x0)/sx)**2)/2), but slightly faster.
    #
    if enablePolynomial is True:
        # Add polynomial
        if a!=0.:
            f += a*x
        if b!=0:
            f += b
    #
    return f


def _gauss2d(x, y, height, x0, y0, sx, sy, a=0., b=0., c=0., enablePolynomial=False):
    """2-d gaussian + 1st order polynomial, with the given parameters.
    The polynomial is in the form a*x + b*y +c."""
    x0 = float(x0) # Will force fx to be float, even if x is int.
    y0 = float(y0) # Will force fy to be float, even if y is int
    #
    fx = x - x0
    # All the following operations on fx are in-place  -> no additional ndarrays created.
    fx /= sx
    fx **= 2
    #
    fy = y - y0
    # All the following operations on fy are in-place  -> no additional ndarrays created.
    fy /= sy
    fy **= 2
    #
    f = fx + fy
    # All the following operations on f are in-place  -> no additional ndarrays created.
    f /= -2
    f = numpy.exp(f)
    f *= height  # like doing f = height*numpy.exp(-(((x-x0)/sx)**2+((y-y0)/sy)**2)/2)
    #
    if enablePolynomial is True:
        # Add polynomial
        if a!=0.:
            f += a*x
        if b!=0:
            f += b*y
        if c!=0:
            f += c
    #
    return f


def _gauss2dRot(x, y, height, x0, y0, sx, sy, theta, a=0., b=0., c=0., enablePolynomial=False):
    """2-d rotated gaussian + 1st order polynomial, with the given parameters.
    The polynomial is in the form a*x + b*y +c."""
    theta = math.radians(theta)
    #
    fx = x*math.cos(theta) + y*math.sin(theta)  # Change reference frame -> x'
    fx -= x0
    fx /= sx
    fx **= 2
    #
    fy = x*math.sin(-theta) + y*math.cos(theta)  # Change reference frame -> y'
    fy -= y0
    fy /= sy
    fy **= 2
    #
    f = height*numpy.exp(-0.5 * (fx+fy))  # like doing f = height*numpy.exp(-(((x'-x0)/sx)**2+((y'-y0)/sy)**2)/2)
    #
    if enablePolynomial is True:
    # Add polynomial
        if a!=0.:
            f += a*x
        if b!=0:
            f += b*y
        if c!=0:
            f += c
    #
    return f
