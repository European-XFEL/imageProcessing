import unittest

import numpy as np

from ..image_exp_running_average import ImageExponentialRunnningAverage


class ImageExpRunningAvg_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.WIDTH = 640  # image width
        cls.HEIGHT = 480  # image height
        cls.PXVALUE = 1024  # image pixel values
        cls.SHAPE = (cls.HEIGHT, cls.WIDTH, 3)
        cls.IMAGE = cls.PXVALUE * np.ones(cls.SHAPE, dtype=np.uint16)
        cls.SPECTRUM = cls.PXVALUE * np.ones((cls.WIDTH,), dtype=np.uint16)

    def test_constructor(self):
        exp_avg = ImageExponentialRunnningAverage()
        self.assertEqual(exp_avg.size, 1)
        self.assertEqual(exp_avg.shape, ())
        self.assertIsNone(exp_avg.mean)

    def test_averaging_method(self):
        exp_avg = ImageExponentialRunnningAverage()

        # Test updating and shape
        exp_avg.append(self.IMAGE, 10)
        self.assertEqual(exp_avg.shape, self.IMAGE.shape)
        exp_avg.append(0.5 * self.IMAGE, 10)
        exp_avg.append(0.5 * self.IMAGE, 10)
        self.assertEqual(exp_avg.shape, self.IMAGE.shape)
        self.assertAlmostEqual(exp_avg.mean[8, 8, 2], 926.72)

        # Test clear average
        exp_avg.clear()
        self.assertIsNone(exp_avg.mean)

        # Test a long averaging run
        exp_avg.clear()
        for ii in range(100):
            exp_avg.append(0.5 * self.IMAGE, 10)
        self.assertAlmostEqual(exp_avg.mean[8, 8, 2], 0.5 * self.PXVALUE, 0.1)
        self.assertEqual(exp_avg.shape, self.IMAGE.shape)

        # Test a very short averaging run
        exp_avg.clear()
        exp_avg.append(0.5 * self.IMAGE, 1)
        exp_avg.append(self.IMAGE, 1)
        exp_avg.append(self.IMAGE, 1)
        self.assertAlmostEqual(exp_avg.mean[8, 8, 2], self.PXVALUE, 0.1)

    def test_spectrum(self):
        exp_avg = ImageExponentialRunnningAverage()

        # Append one image
        exp_avg.append(self.SPECTRUM, 1)
        self.assertEqual(exp_avg.size, 1)
        self.assertEqual(exp_avg.shape, self.SPECTRUM.shape)
        self.assertTrue((exp_avg.mean == self.SPECTRUM).all())
        # Try to append image - must throw!
        with self.assertRaises(ValueError):
            exp_avg.append(self.IMAGE, 1)


if __name__ == '__main__':
    unittest.main()
