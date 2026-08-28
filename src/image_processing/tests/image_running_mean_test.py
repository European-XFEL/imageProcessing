from ..image_running_mean import ImageRunningMean
from .common import GRAY_IMAGE, SPECTRUM


def test_constructor():
    running_mean = ImageRunningMean()
    assert running_mean.size == 0
    assert running_mean.shape == ()
    assert running_mean.runningMean is None


def test_image():
    running_mean = ImageRunningMean()
    # Append one image
    running_mean.append(GRAY_IMAGE)
    assert running_mean.size == 1
    assert running_mean.shape == GRAY_IMAGE.shape
    assert (running_mean.runningMean == GRAY_IMAGE).all()
    # Try to append spectrum - will restart fresh!
    running_mean.append(SPECTRUM)
    assert running_mean.shape == SPECTRUM.shape
    assert running_mean.size == 1
    # Continue with image
    running_mean.append(GRAY_IMAGE)
    assert running_mean.size == 1
    assert running_mean.shape == GRAY_IMAGE.shape
    # Append three more images
    running_mean.append(0.5 * GRAY_IMAGE)
    running_mean.append(0.5 * GRAY_IMAGE)
    running_mean.append(GRAY_IMAGE)
    # Average shall be 0.75*IMAGE
    assert running_mean.size == 4
    assert (running_mean.runningMean == 0.75 * GRAY_IMAGE).all()
    # Pop one image, now average shall be 2/3 * IMAGE
    running_mean.popleft()
    assert running_mean.size == 3
    assert (running_mean.runningMean == 2 / 3 * GRAY_IMAGE).all()
    # Clear average
    running_mean.clear()
    assert running_mean.size == 0
    assert running_mean.shape == ()
    assert running_mean.runningMean is None


def test_spectrum():
    running_mean = ImageRunningMean()
    # Append one image
    running_mean.append(SPECTRUM)
    assert running_mean.size == 1
    assert running_mean.shape == SPECTRUM.shape
    assert (running_mean.runningMean == SPECTRUM).all()
