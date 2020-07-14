from . import Dataset
from . import train

import tensorflow as tf

if __name__ == "__main__":
    image = Dataset.testGenerator('input/test')
    model = train.forward_pass()
    model.load_weights('models/unet.h5')
    results = model.predict_generator(image, 30, verbose = 1)
    Dataset.saveResult('input/results', results)