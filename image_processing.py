#!/usr/bin/env python

__author__="andrea.parenti@xfel.eu"
__date__ ="October 29, 2013, 15:33 AM"
__copyright__="Copyright (c) 2010-2013 European XFEL GmbH Hamburg. All rights reserved."

import numpy
import scipy
import scipy.ndimage
import scipy.stats


def imageSetThreshold(image, threshold):
    """Sets a threshold on an image"""
    if type(image)!=numpy.ndarray:
        return None
    
    return scipy.stats.threshold(image, threshold, None, 0)


def imageXProjection(image):
    """Returns the projection of a 2-D image onto the x-axis"""
    if type(image)!=numpy.ndarray or image.ndim!=2:
        None
    
    return image.sum(axis=0)


def imageYProjection(image):
    """Returns the projection of a 2-D image onto the y-axis"""
    if type(image)!=numpy.ndarray or image.ndim!=2:
        return None
    
    return image.sum(axis=1)


def imageCentreOfMass(image):
    """Returns centre-of-mass and widths of an image (1-D or 2-D)"""
    if type(image)!=numpy.ndarray:
        return None
    
    if image.ndim==1:
        # 1-D image
        
        # Centre-of-mass
        (x0, ) = scipy.ndimage.measurements.center_of_mass(image)
        
        # width
        w  = image
        d  = (numpy.arange(image.shape[0]) - x0)**2
        v  = numpy.dot(w, d)
        sx = numpy.sqrt(v/w.sum())
        
        return (x0, sx)

    elif image.ndim==2:
        # 2-D image

        # Centre-of-mass
        (y0, x0) = scipy.ndimage.measurements.center_of_mass(image)
        
        # x width
        w  = image.sum(axis=0)
        d  = (numpy.arange(image.shape[0]) - x0)**2
        v  = numpy.dot(w, d)
        sx = numpy.sqrt(v/w.sum())
        
        # y width
        w  = image.sum(axis=1)
        d  = (numpy.arange(image.shape[1]) - y0)**2
        v  = numpy.dot(w, d)
        sy = numpy.sqrt(v/w.sum())
        
        return (x0, y0, sx, sy)
    
    else:
        return None


def fitGauss(image, p0=None):
    """Returns gaussian fit parameters of an image (1-D or 2-D)"""
    if type(image)!=numpy.ndarray:
        return None

    if image.ndim==1:
        # 1-D image

        if p0==None:
            # Evaluates initial parameters
            (x0, sx) = imageCentreOfMass(image)
            p0 = (image.max(), x0, sx)

        errFun = lambda p: numpy.ravel(gauss1D(*p)(*numpy.indices(image.shape)) - image)
        p1, success = scipy.optimize.leastsq(errFun, p0)
        return p1, success

    elif image.ndim==2:
        # 2-D image

        if p0==None:
            # Evaluates initial parameters
            (x0, y0, sx, sy) = imageCentreOfMass(image)
            p0 = (image.max(), y0, x0, sy, sx)

        errFun = lambda p: numpy.ravel(gauss2D(*p)(*numpy.indices(image.shape)) - image)
        p1, success = scipy.optimize.leastsq(errFun, p0)
        return p1, success

    else:
        return None


def fitGauss2DRot(image, p0=None):
    """Returns gaussian fit parameters of a 2-D image"""
    if type(image)!=numpy.ndarray or image.ndim!=2:
        return None
    
    if p0==None:
        # Evaluates initial parameters
        (x0, y0, sx, sy) = imageCentreOfMass(image)
        p0 = (image.max(), y0, x0, sy, sx, 0.)
    
    errFun = lambda p: numpy.ravel(gauss2DRot(*p)(*numpy.indices(image.shape)) - image)
    p1, success = scipy.optimize.leastsq(errFun, p0)
    return p1, success


def gauss1D(height, x0, sx):
    """Returns a gaussian function with the given parameters"""
    sx = float(sx)
    return lambda x: height*scipy.exp(-(((x-x0)/sx)**2)/2)


def gauss2D(height, x0, y0, sx, sy):
    """Returns a gaussian function with the given parameters"""
    sx = float(sx)
    sy = float(sy)
    return lambda x,y: height*scipy.exp(-(((x-x0)/sx)**2+((y-y0)/sy)**2)/2)


def gauss2DRot(height, x0, y0, sx, sy, theta):
    """Returns a gaussian function with the given parameters"""
    sx = float(sx)
    sy = float(sy)
    
    theta = numpy.deg2rad(theta)
    x0 = x0 * numpy.cos(theta) - y0 * numpy.sin(theta)
    y0 = x0 * numpy.sin(theta) + y0 * numpy.cos(theta)
    
    def rotgauss(x,y):
        xp = x * numpy.cos(theta) - y * numpy.sin(theta)
        yp = x * numpy.sin(theta) + y * numpy.cos(theta)
        g = height*numpy.exp(-(((xp-x0)/sx)**2+((yp-y0)/sy)**2)/2.)
        return g
    
    return rotgauss
