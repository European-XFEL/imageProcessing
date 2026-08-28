import math

import numpy as np
import pytest

from ..image_processing import (
    _guess1stOrderPolynomial, fitGauss, gauss1d, imageApplyMask,
    imageCentreOfMass, imageFlipAlongX, imageFlipAlongY,
    imagePixelValueFrequencies, imageRotate, imageSelectRegion,
    imageSetThreshold, imageSubtractBackground, imageSumAlongX, imageSumAlongY,
    peakParametersEval, thumbnail)
from .common import (
    GRAY_IMAGE, HEIGHT, IMAGE_STACK, PXVALUE, RGB_IMAGE, SHAPE, SPECTRUM,
    WIDTH)

X1 = 100
X2 = 200
Y1 = 300
Y2 = 400

MASK = np.ones(SHAPE, dtype=np.uint16)
MASK[:X1, :] = 0
BACKGROUND = np.ones(SHAPE, dtype=np.uint16)


def test_masking():
    masked_img = imageApplyMask(GRAY_IMAGE, MASK, True)
    # Verify that a copy of image has been done
    assert masked_img is not GRAY_IMAGE
    assert masked_img.shape == GRAY_IMAGE.shape
    assert masked_img.dtype == GRAY_IMAGE.dtype
    # Verify the result of masking
    assert (masked_img[X1:, :] == PXVALUE).all()  # not masked
    assert (masked_img[:X1, :] == 0).all()  # masked


def test_centre_of_mass():
    x0, sx = imageCentreOfMass(SPECTRUM)
    assert x0 == WIDTH / 2
    assert sx == pytest.approx(WIDTH / math.sqrt(12))

    x0, y0, sx, sy = imageCentreOfMass(GRAY_IMAGE)
    assert x0 == WIDTH / 2
    assert sx == pytest.approx(WIDTH / math.sqrt(12))
    assert y0 == HEIGHT / 2
    assert sy == pytest.approx(HEIGHT / math.sqrt(12))


def test_px_value_frequency():
    px_freq = imagePixelValueFrequencies(GRAY_IMAGE)
    # All pixels have value=PXVALUE
    assert px_freq[PXVALUE].all()
    assert not px_freq[:PXVALUE].any()


def test_region_selection():
    selected_img = imageSelectRegion(GRAY_IMAGE, X1, X2, Y1, Y2, True)
    # Verify that a copy of image has been done
    assert selected_img is not GRAY_IMAGE
    assert selected_img.shape == GRAY_IMAGE.shape
    assert selected_img.dtype == GRAY_IMAGE.dtype
    # Verify the result of selection
    assert (selected_img[Y1:Y2, X1:X2] == PXVALUE).all()
    assert (selected_img[:, :X1] == 0).all()  # not selected
    assert (selected_img[:, X2:] == 0).all()  # not selected
    assert (selected_img[:Y1, :] == 0).all()  # not selected
    assert (selected_img[Y2:, :] == 0).all()  # not selected


def test_region_selection_rgb():
    selected_img = imageSelectRegion(RGB_IMAGE, X1, X2, Y1, Y2, True)
    # Verify that a copy of image has been done
    assert selected_img is not RGB_IMAGE
    assert selected_img.shape == RGB_IMAGE.shape
    assert selected_img.dtype == RGB_IMAGE.dtype
    # Verify the result of selection
    assert (selected_img[Y1:Y2, X1:X2, :] == PXVALUE).all()
    assert (selected_img[:, :X1, :] == 0).all()  # not selected
    assert (selected_img[:, X2:, :] == 0).all()  # not selected
    assert (selected_img[:Y1, :, :] == 0).all()  # not selected
    assert (selected_img[Y2:, :, :] == 0).all()  # not selected


def test_region_selection_stack():
    selected_img = imageSelectRegion(IMAGE_STACK, X1, X2, Y1, Y2, True)
    # Verify that a copy of image has been done
    assert selected_img is not GRAY_IMAGE
    assert selected_img.shape == IMAGE_STACK.shape
    assert selected_img.dtype == IMAGE_STACK.dtype
    # Verify the result of selection
    assert (selected_img[:, Y1:Y2, X1:X2] == PXVALUE).all()
    assert (selected_img[:, :, :X1] == 0).all()  # not selected
    assert (selected_img[:, :, X2:] == 0).all()  # not selected
    assert (selected_img[:, :Y1, :] == 0).all()  # not selected
    assert (selected_img[:, Y2:, :] == 0).all()  # not selected


def test_region_selection_raise():
    with pytest.raises(ValueError):
        # Spectrum (1d data)
        _ = imageSelectRegion(
            np.ones((WIDTH), dtype=np.uint16), X1, X2, Y1, Y2, True)

    with pytest.raises(ValueError):
        # RGB image stack (4d data)
        _ = imageSelectRegion(
            np.ones((10, HEIGHT, WIDTH, 3), dtype=np.uint16),
            X1, X2, Y1, Y2, True)


def test_pixel_threshold():
    image_copy = GRAY_IMAGE.copy()
    image_copy[:X1, :] = 10
    thresh_img = imageSetThreshold(image_copy, X1, True)
    # Verify that a copy of image has been done
    assert thresh_img is not image_copy
    assert thresh_img.shape == image_copy.shape
    assert thresh_img.dtype == image_copy.dtype
    # Verify the result of applying threshold
    assert (thresh_img[X1:, :] == PXVALUE).all()  # above threshold
    assert (thresh_img[:X1, :] == 0).all()  # below threshold


def test_sum():
    img_sumy = imageSumAlongY(GRAY_IMAGE)
    img_sumx = imageSumAlongX(GRAY_IMAGE)
    assert (img_sumy == HEIGHT * PXVALUE).all()
    assert (img_sumx == WIDTH * PXVALUE).all()


def test_sum_rgb():
    img_sumy = imageSumAlongY(RGB_IMAGE)
    img_sumx = imageSumAlongX(RGB_IMAGE)
    assert (img_sumy == HEIGHT * PXVALUE).all()
    assert (img_sumx == WIDTH * PXVALUE).all()


def test_sum_stack():
    img_sumy = imageSumAlongY(IMAGE_STACK)
    img_sumx = imageSumAlongX(IMAGE_STACK)
    assert (img_sumy == HEIGHT * PXVALUE).all()
    assert (img_sumx == WIDTH * PXVALUE).all()


def test_sum_raise():
    for func in (imageSumAlongY, imageSumAlongX):
        with pytest.raises(ValueError):
            # Spectrum (1d data)
            _ = func(np.ones((WIDTH), dtype=np.uint16))

        with pytest.raises(ValueError):
            # RGB image stack (4d data)
            _ = func(np.ones((10, HEIGHT, WIDTH, 3), dtype=np.uint16))


def test_subtract_bkg():
    subtracted_img = imageSubtractBackground(GRAY_IMAGE, BACKGROUND, True)
    # Verify that a copy of image has been done
    assert subtracted_img is not GRAY_IMAGE
    assert subtracted_img.shape == GRAY_IMAGE.shape
    assert subtracted_img.dtype == GRAY_IMAGE.dtype
    # Verify the result of background subtraction
    assert (subtracted_img == PXVALUE - 1).all()


def test_peak_param():
    x = np.arange(WIDTH)
    x0 = 350  # peak position
    sx = 20  # variance
    sigma_to_fwhm = 2 * math.sqrt(2 * math.log(2))
    ampl, maxPos, fwhm = peakParametersEval(gauss1d(x, PXVALUE, x0, sx))

    assert ampl == pytest.approx(PXVALUE)
    assert maxPos == pytest.approx(x0)
    assert fwhm == pytest.approx(sx * sigma_to_fwhm, abs=1)

    with pytest.raises(ValueError):
        # Max is on last pixel
        peakParametersEval(x)


def test_inplace_algorithms():
    # Check that algorithms are applied in-place
    image_copy = np.copy(GRAY_IMAGE)
    assert imageApplyMask(image_copy, MASK) is image_copy
    assert imageSubtractBackground(image_copy, BACKGROUND) is image_copy
    assert imageSelectRegion(image_copy, X1, X2, Y1, Y2) is image_copy
    assert imageSetThreshold(image_copy, X1) is image_copy


def test_thumbnail():
    def test_helper(rectangle, expected_shape, resample):
        thumb_img = thumbnail(GRAY_IMAGE, rectangle, resample=resample)
        assert thumb_img.shape == expected_shape
        return thumb_img

    # Test that dimensions are handled as expected

    # A simple one: resized image fits exactly in canvas
    test_helper(
        (HEIGHT // 2, WIDTH // 2), (HEIGHT // 2, WIDTH // 2), True)
    test_helper(
        (HEIGHT // 2, WIDTH // 2), (HEIGHT // 2, WIDTH // 2), False)

    # x fits exactly, y doesn't
    test_helper((130, 192), (108, 192), True)
    test_helper((130, 192), (108, 192), False)

    # y fits exactly, x doesn't
    test_helper((108, 200), (108, 192), True)
    test_helper((108, 200), (108, 192), False)

    # This triggers padding (scaling factor is 7)
    thumb_img = test_helper((158, 300), (155, 275), False)

    # Check that padding worked as expected
    assert (thumb_img[:-1, :-1] == PXVALUE * np.ones((154, 274))).all()
    assert (thumb_img[:, -1:] == np.zeros((155, 1))).all()
    assert (thumb_img[-1:, :] == np.zeros((1, 275))).all()

    # Test image content...
    image = np.arange(1, 25, dtype=np.uint16).reshape(4, 6)

    # ... with averaged binning (resample)
    thumb_img = thumbnail(image, (3, 3), resample=True)
    assert thumb_img.shape == (2, 3)
    assert (thumb_img == [[4.5, 6.5, 8.5], [16.5, 18.5, 20.5]]).all()

    # ... w/o averaging (plain downsampling)
    thumb = thumbnail(image, (3, 3))
    assert thumb_img.shape == (2, 3)
    assert (thumb == [[1, 3, 5], [13, 15, 17]]).all()

    # Image already fits in canvas
    thumb = thumbnail(image, (8, 6))
    assert (thumb == image).all()


def test_fit_gauss():
    x = np.arange(WIDTH)
    x0 = 350  # peak position
    sx = 20  # variance
    a = 0.1
    b = 120.0
    ampl = PXVALUE
    res, _, _ = fitGauss(gauss1d(x, ampl, x0, sx), enablePolynomial=False)

    assert res[0] == pytest.approx(ampl, abs=10)
    assert res[1] == pytest.approx(x0)
    assert res[2] == pytest.approx(sx, abs=1)

    curve = gauss1d(x, ampl, x0, sx, a=a, b=b, enablePolynomial=True)
    res, _, _ = fitGauss(curve, enablePolynomial=True)
    assert res[0] == pytest.approx(ampl, abs=10)
    assert res[1] == pytest.approx(x0)
    assert res[2] == pytest.approx(sx, abs=1)


def test_guess_polynomial():
    x = np.arange(WIDTH)
    x0 = 350  # peak position
    sx = 20  # variance
    a = 0.3
    b = 1.0
    curve = gauss1d(x, PXVALUE, x0, sx, a=a, b=b, enablePolynomial=True)
    a0, b0 = _guess1stOrderPolynomial(curve)
    a0 == pytest.approx(a)
    b0 == pytest.approx(b)

    a1, b1 = _guess1stOrderPolynomial(x)
    a1 == pytest.approx(1)
    b1 == pytest.approx(0)

    a1, b1 = _guess1stOrderPolynomial(np.flip(x))
    a1 == pytest.approx(-1)
    b1 == pytest.approx(WIDTH)


def test_rotate():
    img = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        dtype=np.uint32)

    rot = imageRotate(img)
    res = (
        rot == np.array(
            [[4, 8, 12], [3, 7, 11], [2, 6, 10], [1, 5, 9]], dtype=np.uint32))

    assert res.all()

    rot = imageRotate(img, 180)
    res = (
        rot == np.array(
            [[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]], dtype=np.uint32))

    assert res.all()

    rot = imageRotate(img, 270)
    res = (
        rot == np.array(
            [[9, 5, 1], [10, 6, 2], [11, 7, 3], [12, 8, 4]], dtype=np.uint32))

    assert res.all()

    rot = imageRotate(img, 360)
    res = (rot == img)

    assert res.all()

    with pytest.raises(ValueError):
        imageRotate(img, 123)


def test_flip():
    img = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint32)

    # Test flip along X

    flipped = imageFlipAlongX(img)

    res = (
        flipped == np.array(
            [[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]], dtype=np.uint32))

    assert res.all()

    flipped_twice = imageFlipAlongX(flipped)

    res = (flipped_twice == img)

    assert res.all()

    # Test flip along Y

    flipped = imageFlipAlongY(img)

    res = (flipped == np.array(
        [[9, 10, 11, 12], [5, 6, 7, 8], [1, 2, 3, 4]], dtype=np.uint32))

    assert res.all()

    flipped_twice = imageFlipAlongY(flipped)

    res = (flipped_twice == img)

    assert res.all()
