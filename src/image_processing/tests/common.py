import numpy as np

WIDTH = 1920  # image width
HEIGHT = 1080  # image height
PXVALUE = 1024  # image pixel values

SHAPE = (HEIGHT, WIDTH)
GRAY_IMAGE = PXVALUE * np.ones(SHAPE, dtype=np.uint16)
RGB_IMAGE = PXVALUE * np.ones((HEIGHT, WIDTH, 3), dtype=np.uint16)
IMAGE_STACK = PXVALUE * np.ones((10, HEIGHT, WIDTH), dtype=np.uint16)
SPECTRUM = PXVALUE * np.ones((WIDTH,), dtype=np.uint16)
