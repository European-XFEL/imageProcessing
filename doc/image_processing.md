# The `image_processing` submodule

Import the image-processing functions with:

```python
from image_processing.image_processing import *
```

Many functions modify the input image by default. To avoid this, use the
`copy=True` option.

## Available functions

#### `imagePixelValueFrequencies(image)`
Calls `numpy.bincount` and returns one-dimensional pixel-value frequencies.
The input image must have an integer dtype.

#### `imageSetThreshold(image, threshold, copy=False)`
Sets pixels below `threshold` to zero.

#### `imageSubtractBackground(image, background, copy=False)`
Subtracts `background` from `image`; the image dtype should be signed since
the subtraction can result in negative values.


#### `imageApplyMask(image, mask, copy=False)`
Applies `mask`; mask and image must have the same shape. Values less than or
equal to zero mask pixels out.

#### `imageSelectRegion(image, x1, x2, y1, y2, copy=False)`
Sets everything outside the rectangular region to zero.

#### `imageSumAlongY(image)` / `imageSumAlongX(image)`
Integrate the image along the Y or X axis.

#### `imageCentreOfMass(image)`
Returns centre of mass and standard deviation: `(x0, sx)` for 1D images and
`(x0, y0, sx, sy)` for 2D images.

#### `fitGauss(image, p0=None, enablePolynomial=False)`
Returns Gaussian-fit parameters for one- or two-dimensional images. An initial
estimate may be supplied through `p0`; a first-order polynomial can be added.
For 1D images, return values are `(A, x0, sx, covariance, error)` or
`(A, x0, sx, a, c, covariance, error)`. For 2D images, they are
`(A, x0, y0, sx, sy, covariance, error)` or
`(A, x0, y0, sx, sy, a, b, c, covariance, error)`.
The fit is done with the help of the `scipy.optimize.leastsq` function, with `full_output` option.

#### `fitGauss2DRot(image, p0=None, enablePolynomial=False)`
Returns Gaussian-fit parameters for a 2D image, including rotation. An initial
estimate may be supplied through `p0`; a first-order polynomial can be added.
Return values include `theta`, covariance, and error, with polynomial
coefficients included when requested.

#### `fitSech2(image, p0=None, enablePolynomial=False)`
Returns squared [hyperbolic-secant](http://mathworld.wolfram.com/HyperbolicSecant.html)
fit parameters for a 1D image.

#### `peakParametersEval(img)`
Evaluates `(maxValue, maxPosition, FWHM)` for 1D distributions.
The parameters are calculated with no assumption on the peak shape, the
only requirement is having a single maximum.


#### `thumbnail(image, canvas, resample=False)`
Downscales an image by an integer factor to fit the canvas while preserving the
X–Y ratio. If needed, it pads with zero when `resample=False`, or with edge
values when `resample=True`; an image that already fits is returned unchanged.
