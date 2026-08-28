import pytest

from ..image_standard_mean import ImageStandardMean
from .common import GRAY_IMAGE, SPECTRUM


def test_constructor():
    std_mean = ImageStandardMean()
    assert std_mean.size == 0
    assert std_mean.shape == ()
    assert std_mean.mean is None


def test_image():
    std_mean = ImageStandardMean()
    # Append one image
    std_mean.append(GRAY_IMAGE)
    assert std_mean.size == 1
    assert std_mean.shape == GRAY_IMAGE.shape
    assert (std_mean.mean == GRAY_IMAGE).all()
    # Try to append spectrum - must throw!
    with pytest.raises(ValueError):
        std_mean.append(SPECTRUM)
    # Append three more images
    std_mean.append(0.5 * GRAY_IMAGE)
    std_mean.append(0.5 * GRAY_IMAGE)
    std_mean.append(GRAY_IMAGE)
    # Average shall be 0.75*IMAGE
    assert std_mean.size == 4
    assert (std_mean.mean == 0.75 * GRAY_IMAGE).all()
    # Clear average
    std_mean.clear()
    assert std_mean.size == 0
    assert std_mean.shape == ()
    assert std_mean.mean is None


def test_spectrum():
    std_mean = ImageStandardMean()
    # Append one image
    std_mean.append(SPECTRUM)
    assert std_mean.size == 1
    assert std_mean.shape == SPECTRUM.shape
    assert (std_mean.mean == SPECTRUM).all()
    # Try to append image - must throw!
    with pytest.raises(ValueError):
        std_mean.append(GRAY_IMAGE)
