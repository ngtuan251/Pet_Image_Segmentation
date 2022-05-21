import tensorflow as tf
import numpy as np
from keras.models import load_model
from utils.plot_image import display

from utils.config import Config

from configs.config import CFG

class UnetInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.image_size = self.config.data.image_size
        self.model_path = 'D:\Deep-Learning-In-Production-master\Deploy\checkpoints\model.03-0.2907.h5'
        self.model = load_model(self.model_path)
        # self.predict = self.model.signatures["serving_default"]

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32)
        # return image

    def infer(self, image=None):
        image_ori = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = self.preprocess(image_ori)
        shape = image_tensor.shape
        print(shape)
        image_tensor = tf.reshape(
            image_tensor, [1, shape[0], shape[1], shape[2]]
        )
        print(image_tensor.shape)
        pred = self.model.predict(image_tensor)
        print(pred.shape)
        print(image_tensor[0].shape)
        display([image_ori, pred[0]])
        # pred = pred.numpy().tolist()
        # return {'segmentation_output': pred}
        return pred

