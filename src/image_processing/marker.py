import math

import cv2
import numpy as np


def rectangular_marker(image, center, shape, color=None, thickness=5, angle=0):
    """Superimpose a rectangular marker to the input image.

    :param image: the input image (it will be modified in-place)
    :param center: the (x, y) coordinates of the rectangle's center
    :param shape: the shape of the rectangle (width, height)
    :param color: the "color" of the rectangle to be superimposed (by default
    image.min())
    :param thickness: the thickness of the rectangle
    :parameter angle: the clockwise rotation angle, in degrees
    """
    if color is None:
        color = int(image.min())

    x0 = center[0]
    y0 = center[1]

    w2 = shape[0] // 2  # half width
    h2 = shape[1] // 2  # half height

    if angle % 180 == 0:
        pt1 = (x0 - w2, y0 - h2)  # upper-left corner
        pt2 = (x0 + w2, y0 + h2)  # lower right corner
        cv2.rectangle(image, pt1, pt2, color, thickness)
    elif angle % 90 == 0:
        pt1 = (x0 - h2, y0 - w2)  # upper-left corner
        pt2 = (x0 + h2, y0 + w2)  # loer-right corner
        cv2.rectangle(image, pt1, pt2, color, thickness)
    else:
        alpha = angle / 180 * math.pi
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        dx1 = int(w2 * cos_a)
        dx2 = int(h2 * sin_a)
        dy1 = int(h2 * cos_a)
        dy2 = int(w2 * sin_a)

        pt1 = (x0 + dx1 + dx2, y0 - dy1 + dy2)  # upper-right corner
        pt2 = (x0 - dx1 + dx2, y0 - dy1 - dy2)  # upper-left corner
        pt3 = (x0 - dx1 - dx2, y0 + dy1 - dy2)  # lower-left corner
        pt4 = (x0 + dx1 - dx2, y0 + dy1 + dy2)  # lower-right corner

        rect = np.array([[pt1, pt2, pt3, pt4]], np.int32)

        cv2.polylines(image, rect, True, color, thickness)


def circular_marker(image, center, radius, color=None, thickness=5):
    """Superimpose a circular marker to the input image.

    :param image: the input image (it will be modified in-place)
    :param center: the (x, y) coordinates of the circle's center
    :param radius: the radius of the circle
    :param color: the "color" of the circle to be superimposed (by default
    image.min())
    :param thickness: the thickness of the circle
    """
    if color is None:
        color = int(image.min())

    cv2.circle(image, center, radius, color, thickness)


def ellipsoidal_marker(image, center, axes, color=None, thickness=5, angle=0):
    """Superimpose an ellipsoidal marker to the input image.

    :param image: the input image (it will be modified in-place)
    :param center: the (x, y) coordinates of the ellipse's center
    :param axes: the axes of the ellipse (horizonal axis, vertical axis)
    :param color: the "color" of the ellipse to be superimposed (by default
    image.min())
    :param thickness: the thickness of the rectangle
    :parameter angle: the clockwise rotation angle, in degrees
    """
    if color is None:
        color = int(image.min())

    cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)


def marker(image, marker_type, center, shape, color=None, thickness=5,
           angle=0):
    """Superimpose a marker to the input image.

    :param image: the input image (it will be modified in-place)
    :param marker_type: the marker type: 'rectangle', 'ellipse'
    :param center: the (x, y) coordinates of the ellipse's center
    :param shape: the dimensions of the marker (width, height)
    :param color: the "color" of the ellipse to be superimposed (by default
    image.min())
    :param thickness: the thickness of the rectangle
    :parameter angle: the clockwise rotation angle, in degrees
    """
    if marker_type == 'rectangle':
        rectangular_marker(image, center, shape, color, thickness, angle)

    elif marker_type == 'ellipse':
        if shape[0] == shape[1]:
            radius = shape[0] // 2
            circular_marker(image, center, radius, color, thickness)
        else:
            axes = (shape[0] // 2, shape[1] // 2)
            ellipsoidal_marker(image, center, axes, color, thickness, angle)

    else:
        raise ValueError(f"{marker_type} type is not supported")
