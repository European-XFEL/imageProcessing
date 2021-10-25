import math

from .marker import rectangular_marker


def crosshair(image, center, ext_size, int_size=0, color=None, thickness=5,
              angle=0):
    """Superimpose a cross-hair to the input image.

    :param image: the input image (it will be modified in-place)
    :param center: the (x, y) coordinates of the cross-hair's center
    :param ext_size: the size of the cross-hair
    :param int_size: the size of the transparent part of the cross-hair
    :param color: the "color" of the cross-hair to be superimposed (by default
    image.min())
    :param thickness: the thickness of the cross-hair
    :parameter angle: the clockwise rotation angle, in degrees
    """
    if color is None:
        color = int(image.min())

    if int_size > 0 and ext_size > int_size:
        w = ext_size - int_size
        h = thickness // 2
        d = (ext_size + int_size) // 2
        x = center[0]
        y = center[1]

        alpha = angle / 180 * math.pi
        d1 = int(d * math.cos(alpha))
        d2 = int(d * math.sin(alpha))

        rectangular_marker(image, (x + d1, y + d2), (w, h), color, h, angle)
        rectangular_marker(image, (x - d1, y - d2), (w, h), color, h, angle)
        rectangular_marker(image, (x - d2, y + d1), (h, w), color, h, angle)
        rectangular_marker(image, (x + d2, y - d1), (h, w), color, h, angle)

    else:
        rectangular_marker(image, center, (thickness // 2, 2 * ext_size),
                           color, thickness // 2, angle)
        rectangular_marker(image, center, (2 * ext_size, thickness // 2),
                           color, thickness // 2, angle)
