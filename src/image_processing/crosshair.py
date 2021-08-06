from .marker import rectangular_marker


def crosshair(image, center, ext_size, int_size=0, ext_color=None,
              int_color=None, thickness=5, angle=0):
    """Superimpose a cross-hair to the input image.

    :param image: the input image (it will be modified in-place)
    :param center: the (x, y) coordinates of the cross-hair's center
    :param ext_size: the size of the external part of the cross-hair
    :param int_size: the size of the internal part of the cross-hair
    :param ext_color: the "color" of the external cross-hair to be superimposed
    (by default image.min())
    :param int_color: the "color" of the internal cross-hair to be superimposed
    (by default image.max())
    :param thickness: the thickness of the cross-hair
    :parameter angle: the clockwise rotation angle, in degrees
    """
    if ext_color is None:
        ext_color = int(image.min())

    if int_color is None:
        int_color = int(image.max())

    rectangular_marker(image, center, (thickness // 2, 2 * ext_size),
                       ext_color, thickness // 2, angle)
    rectangular_marker(image, center, (2 * ext_size, thickness // 2),
                       ext_color, thickness // 2, angle)

    if int_size > 0 and int_color != ext_color:
        rectangular_marker(image, center, (thickness // 2, 2 * int_size),
                           int_color, thickness // 2, angle)
        rectangular_marker(image, center, (2 * int_size, thickness // 2),
                           int_color, thickness // 2, angle)
