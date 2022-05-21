import unittest
from executor.unet_inferrer import UnetInferrer
from PIL import Image
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_infer(self):
        image = np.asarray(Image.open('D:\Deep-Learning-In-Production-master\Deploy\con_ngua.jpg')).astype(np.float32)
        unet_inferrer = UnetInferrer()
        unet_inferrer.infer(image)