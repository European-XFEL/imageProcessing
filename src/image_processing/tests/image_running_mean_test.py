import unittest

import numpy as np

from ..image_running_mean import ImageRunningMean


class ImageRunningMean_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.WIDTH = 1920  # image width
        cls.HEIGHT = 1080  # image height
        cls.PXVALUE = 1000  # image pixel values
        cls.SHAPE = (cls.HEIGHT, cls.WIDTH)
        cls.IMAGE = cls.PXVALUE * np.ones(cls.SHAPE, dtype=np.uint16)
        cls.SPECTRUM = cls.PXVALUE * np.ones((cls.WIDTH,), dtype=np.uint16)

    def test_constructor(self):
        running_mean = ImageRunningMean()
        self.assertEqual(running_mean.size, 0)
        self.assertEqual(running_mean.shape, ())
        self.assertIsNone(running_mean.runningMean)

    def test_image(self):
        running_mean = ImageRunningMean()
        # Append one image
        running_mean.append(self.IMAGE)
        self.assertEqual(running_mean.size, 1)
        self.assertEqual(running_mean.shape, self.IMAGE.shape)
        self.assertTrue((running_mean.runningMean == self.IMAGE).all())
        # Try to append spectrum - will restart fresh!
        running_mean.append(self.SPECTRUM)
        self.assertEqual(running_mean.shape, self.SPECTRUM.shape)
        self.assertEqual(running_mean.size, 1)
        # Continue with image
        running_mean.append(self.IMAGE)
        self.assertEqual(running_mean.size, 1)
        self.assertEqual(running_mean.shape, self.IMAGE.shape)
        # Append three more images
        running_mean.append(0.5 * self.IMAGE)
        running_mean.append(0.5 * self.IMAGE)
        running_mean.append(self.IMAGE)
        # Average shall be 0.75*self.IMAGE
        self.assertEqual(running_mean.size, 4)
        self.assertTrue((running_mean.runningMean == 0.75 * self.IMAGE).all())
        # Pop one image, now average shall be 2/3*self.IMAGE
        running_mean.popleft()
        self.assertEqual(running_mean.size, 3)
        self.assertTrue((running_mean.runningMean == 2/3 * self.IMAGE).all())
        # Clear average
        running_mean.clear()
        self.assertEqual(running_mean.size, 0)
        self.assertEqual(running_mean.shape, ())
        self.assertIsNone(running_mean.runningMean)

    def test_spectrum(self):
        running_mean = ImageRunningMean()
        # Append one image
        running_mean.append(self.SPECTRUM)
        self.assertEqual(running_mean.size, 1)
        self.assertEqual(running_mean.shape, self.SPECTRUM.shape)
        self.assertTrue((running_mean.runningMean == self.SPECTRUM).all())


if __name__ == '__main__':
    unittest.main()
