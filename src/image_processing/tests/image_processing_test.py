import unittest

import math
import numpy as np

from ..image_processing import (
    gauss1d, imageApplyMask, imageCentreOfMass, imagePixelValueFrequencies,
    imageSelectRegion, imageSetThreshold, imageSumAlongX, imageSumAlongY,
    imageSubtractBackground, peakParametersEval
)


class ImageProcessing_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.WIDTH = 1920  # image width
        cls.HEIGHT = 1080  # image height
        cls.PXVALUE = 1000  # image pixel values
        cls.SHAPE = (cls.HEIGHT, cls.WIDTH)
        cls.X1 = 100
        cls.X2 = 200
        cls.Y1 = 300
        cls.Y2 = 400
        cls.IMAGE = cls.PXVALUE * np.ones(cls.SHAPE, dtype=np.uint16)
        cls.SPECTRUM = cls.PXVALUE * np.ones((cls.WIDTH,), dtype=np.uint16)
        cls.MASK = np.ones(cls.SHAPE, dtype=np.uint16)
        cls.MASK[:cls.X1, :] = 0
        cls.BACKGROUND = np.ones(cls.SHAPE, dtype=np.uint16)

    def test_masking(self):
        masked_img = imageApplyMask(self.IMAGE, self.MASK, True)
        # Verify that a copy of image has been done
        self.assertTrue(masked_img is not self.IMAGE)
        self.assertEqual(masked_img.shape, self.IMAGE.shape)
        self.assertEqual(masked_img.dtype, self.IMAGE.dtype)
        # Verify the result of masking
        # not masked:
        self.assertTrue((masked_img[self.X1:, :] == self.PXVALUE).all())
        self.assertTrue((masked_img[:self.X1, :] == 0).all())  # masked

    def test_centre_of_mass(self):
        x0, sx = imageCentreOfMass(self.SPECTRUM)
        self.assertEqual(x0, self.WIDTH / 2)
        self.assertAlmostEqual(sx, self.WIDTH / math.sqrt(12), places=3)

        x0, y0, sx, sy = imageCentreOfMass(self.IMAGE)
        self.assertEqual(x0, self.WIDTH / 2)
        self.assertAlmostEqual(sx, self.WIDTH / math.sqrt(12), places=3)
        self.assertEqual(y0, self.HEIGHT / 2)
        self.assertAlmostEqual(sy, self.HEIGHT / math.sqrt(12), places=3)

    def test_px_value_frequency(self):
        px_freq = imagePixelValueFrequencies(self.IMAGE)
        # all pixels have value=self.PXVALUE
        self.assertTrue(px_freq[self.PXVALUE].all())
        self.assertFalse(px_freq[:self.PXVALUE].any())

    def test_region_selection(self):
        selected_img = imageSelectRegion(self.IMAGE, self.X1, self.X2,
                                         self.Y1, self.Y2, True)
        # Verify that a copy of image has been done
        self.assertTrue(selected_img is not self.IMAGE)
        self.assertEqual(selected_img.shape, self.IMAGE.shape)
        self.assertEqual(selected_img.dtype, self.IMAGE.dtype)
        # Verify the result of selection
        self.assertTrue((selected_img[self.Y1:self.Y2, self.X1:self.X2]
                         == self.PXVALUE).all())
        self.assertTrue((selected_img[:, :self.X1] == 0).all())  # not selected
        self.assertTrue((selected_img[:, self.X2:] == 0).all())  # not selected
        self.assertTrue((selected_img[:self.Y1, :] == 0).all())  # not selected
        self.assertTrue((selected_img[self.Y2:, :] == 0).all())  # not selected

    def test_pixel_threshold(self):
        image_copy = self.IMAGE.copy()
        image_copy[:self.X1, :] = 10
        thresh_img = imageSetThreshold(image_copy, self.X1, True)
        # Verify that a copy of image has been done
        self.assertTrue(thresh_img is not image_copy)
        self.assertEqual(thresh_img.shape, image_copy.shape)
        self.assertEqual(thresh_img.dtype, image_copy.dtype)
        # Verify the result of applying threshold
        # above threshold:
        self.assertTrue((thresh_img[self.X1:, :] == self.PXVALUE).all())
        self.assertTrue((thresh_img[:self.X1, :] == 0).all())  # below thresh.

    def test_sum(self):
        img_sumy = imageSumAlongY(self.IMAGE)
        img_sumx = imageSumAlongX(self.IMAGE)
        self.assertTrue((img_sumy == self.HEIGHT * self.PXVALUE).all())
        self.assertTrue((img_sumx == self.WIDTH * self.PXVALUE).all())

    def test_subtract_bkg(self):
        subtracted_img = imageSubtractBackground(self.IMAGE, self.BACKGROUND,
                                                 True)
        # Verify that a copy of image has been done
        self.assertTrue(subtracted_img is not self.IMAGE)
        self.assertEqual(subtracted_img.shape, self.IMAGE.shape)
        self.assertEqual(subtracted_img.dtype, self.IMAGE.dtype)
        # Verify the result of background subtraction
        self.assertTrue((subtracted_img == self.PXVALUE - 1).all())

    def test_peak_param(self):
        x = np.arange(self.WIDTH)
        x0 = 350  # peak position
        sx = 20  # variance
        sigma_to_fwhm = 2 * math.sqrt(2 * math.log(2))
        ampl, maxPos, fwhm = peakParametersEval(
            gauss1d(x, self.PXVALUE, x0, sx))
        self.assertAlmostEqual(ampl, self.PXVALUE)
        self.assertAlmostEqual(maxPos, x0)
        self.assertAlmostEqual(fwhm, sx * sigma_to_fwhm, delta=1)

    def test_inplace_algorithms(self):
        # Check that algorithms are applied in-place
        image_copy = np.copy(self.IMAGE)
        self.assertTrue(imageApplyMask(image_copy, self.MASK) is image_copy)
        self.assertTrue(imageSubtractBackground(image_copy, self.BACKGROUND)
                        is image_copy)
        self.assertTrue(imageSelectRegion(image_copy, self.X1, self.X2,
                                          self.Y1, self.Y2) is image_copy)
        self.assertTrue(imageSetThreshold(image_copy, self.X1) is image_copy)


if __name__ == '__main__':
    unittest.main()
