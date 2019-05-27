import math
import unittest

import numpy as np

from ..image_processing import (
    thumbnail, gauss1d, fitGauss, _guess1stOrderPolynomial, imageApplyMask,
    imageCentreOfMass, imagePixelValueFrequencies, imageSelectRegion,
    imageSetThreshold, imageSumAlongX, imageSumAlongY, imageSubtractBackground,
    peakParametersEval
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

    def test_thumbnail(self):
        def test_helper(rectangle, expected_shape, resample):
            thumb_img = thumbnail(self.IMAGE, rectangle, resample=resample)
            self.assertEqual(thumb_img.shape, expected_shape)
            return thumb_img

        # test that dimensions are handled as expected

        # a simple one: resized image fits exactly in canvas
        test_helper((self.HEIGHT // 2, self.WIDTH // 2),
                    (self.HEIGHT // 2, self.WIDTH // 2), True)
        test_helper((self.HEIGHT // 2, self.WIDTH // 2),
                    (self.HEIGHT // 2, self.WIDTH // 2), False)

        # x fits exactly, y doesn't
        test_helper((130, 192), (108, 192), True)
        test_helper((130, 192), (108, 192), False)

        # y fits exactly, x doesn't
        test_helper((108, 200), (108, 192), True)
        test_helper((108, 200), (108, 192), False)

        # this triggers padding (scaling factor is 7)
        thumb_img = test_helper((158, 300), (155, 275), False)

        # check that padding worked as expected
        self.assertTrue((thumb_img[:-1, :-1] ==
                         self.PXVALUE * np.ones((154, 274))).all())
        self.assertTrue((thumb_img[:, -1:] == np.zeros((155, 1))).all())
        self.assertTrue((thumb_img[-1:, :] == np.zeros((1, 275))).all())

        # test image content...
        image = np.arange(1, 25, dtype=np.uint16).reshape(4, 6)

        # ... with averaged binning (resample)
        thumb_img = thumbnail(image, (3, 3), resample=True)
        self.assertEqual(thumb_img.shape, (2, 3))
        self.assertTrue(
            (thumb_img == [[4.5, 6.5, 8.5], [16.5, 18.5, 20.5]]).all())

        # ... w/o averaging (plain downsampling)
        thumb = thumbnail(image, (3, 3))
        self.assertEqual(thumb_img.shape, (2, 3))
        self.assertTrue((thumb == [[1, 3, 5], [13, 15, 17]]).all())

        # image already fits in canvas
        thumb = thumbnail(image, (8, 6))
        self.assertTrue((thumb == image).all())

    def test_fit_gauss(self):
        x = np.arange(self.WIDTH)
        x0 = 350  # peak position
        sx = 20  # variance
        a = 0.1
        b = 120.0
        ampl = self.PXVALUE
        res, cov, err = fitGauss(gauss1d(x, ampl, x0, sx),
                                 enablePolynomial=False)

        self.assertAlmostEqual(res[0], ampl, delta=10)
        self.assertAlmostEqual(res[1], x0)
        self.assertAlmostEqual(res[2], sx, delta=1)

        curve = gauss1d(x, ampl, x0, sx, a=a, b=b,
                           enablePolynomial=True)
        res, cov, err = fitGauss(curve, enablePolynomial=True)
        self.assertAlmostEqual(res[0], ampl, delta=10)
        self.assertAlmostEqual(res[1], x0 )
        self.assertAlmostEqual(res[2], sx, delta=1)

    def test_guess_polynomial(self):
        x = np.arange(self.WIDTH)
        x0 = 350  # peak position
        sx = 20  # variance
        a = 0.3
        b = 1.0
        curve = gauss1d(x, self.PXVALUE, x0, sx, a=a, b=b,
                enablePolynomial=True)
        a0, b0 = _guess1stOrderPolynomial(curve)
        self.assertAlmostEqual(a0, a, delta=0.01)
        self.assertAlmostEqual(b0, b, delta=0.01)


if __name__ == '__main__':
    unittest.main()
