import unittest

import numpy as np

from ..image_standard_mean import ImageStandardMean


class ImageStandardMean_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.WIDTH = 1920  # image width
        cls.HEIGHT = 1080  # image height
        cls.PXVALUE = 1000  # image pixel values
        cls.SHAPE = (cls.HEIGHT, cls.WIDTH)
        cls.IMAGE = cls.PXVALUE * np.ones(cls.SHAPE, dtype=np.uint16)
        cls.SPECTRUM = cls.PXVALUE * np.ones((cls.WIDTH,), dtype=np.uint16)

    def test_constructor(self):
        std_mean = ImageStandardMean()
        self.assertEqual(std_mean.size, 0)
        self.assertEqual(std_mean.shape, ())
        self.assertIsNone(std_mean.mean)

    def test_image(self):
        std_mean = ImageStandardMean()
        # Append one image
        std_mean.append(self.IMAGE)
        self.assertEqual(std_mean.size, 1)
        self.assertEqual(std_mean.shape, self.IMAGE.shape)
        self.assertTrue((std_mean.mean == self.IMAGE).all())
        # Try to append spectrum - must throw!
        with self.assertRaises(ValueError):
            std_mean.append(self.SPECTRUM)
        # Append three more images
        std_mean.append(0.5 * self.IMAGE)
        std_mean.append(0.5 * self.IMAGE)
        std_mean.append(self.IMAGE)
        # Average shall be 0.75*self.IMAGE
        self.assertEqual(std_mean.size, 4)
        self.assertTrue((std_mean.mean == 0.75 * self.IMAGE).all())
        # Clear average
        std_mean.clear()
        self.assertEqual(std_mean.size, 0)
        self.assertEqual(std_mean.shape, ())
        self.assertIsNone(std_mean.mean)

    def test_spectrum(self):
        std_mean = ImageStandardMean()
        # Append one image
        std_mean.append(self.SPECTRUM)
        self.assertEqual(std_mean.size, 1)
        self.assertEqual(std_mean.shape, self.SPECTRUM.shape)
        self.assertTrue((std_mean.mean == self.SPECTRUM).all())
        # Try to append image - must throw!
        with self.assertRaises(ValueError):
            std_mean.append(self.IMAGE)


if __name__ == '__main__':
    unittest.main()
