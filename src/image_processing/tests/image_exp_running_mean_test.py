import pytest

from ..image_exp_running_average import ImageExponentialRunnningAverage
from .common import PXVALUE, RGB_IMAGE, SPECTRUM


def test_constructor():
    exp_avg = ImageExponentialRunnningAverage()
    assert exp_avg.size == 1
    assert exp_avg.shape == ()
    assert exp_avg.mean is None


def test_averaging_method():
    exp_avg = ImageExponentialRunnningAverage()

    # Test updating and shape
    exp_avg.append(RGB_IMAGE, 10)
    assert exp_avg.shape == RGB_IMAGE.shape
    exp_avg.append(0.5 * RGB_IMAGE, 10)
    exp_avg.append(0.5 * RGB_IMAGE, 10)
    assert exp_avg.shape == RGB_IMAGE.shape
    assert exp_avg.mean[8, 8, 2] == pytest.approx(926.72)

    # Test clear average
    exp_avg.clear()
    assert exp_avg.mean is None

    # Test a long averaging run
    exp_avg.clear()
    for ii in range(100):
        exp_avg.append(0.5 * RGB_IMAGE, 10)
    assert exp_avg.mean[8, 8, 2] == pytest.approx(0.5 * PXVALUE)
    assert exp_avg.shape == RGB_IMAGE.shape

    # Test a very short averaging run
    exp_avg.clear()
    exp_avg.append(0.5 * RGB_IMAGE, 1)
    exp_avg.append(RGB_IMAGE, 1)
    exp_avg.append(RGB_IMAGE, 1)
    assert exp_avg.mean[8, 8, 2] == pytest.approx(PXVALUE)


def test_spectrum():
    exp_avg = ImageExponentialRunnningAverage()

    # Append one image
    exp_avg.append(SPECTRUM, 1)
    assert exp_avg.size == 1
    assert exp_avg.shape == SPECTRUM.shape
    assert (exp_avg.mean == SPECTRUM).all()
    # Try to append image - must throw!
    with pytest.raises(ValueError):
        exp_avg.append(RGB_IMAGE, 1)
