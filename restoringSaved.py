from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow.python.saved_model import tag_constants

def testingVector():
    im1 = Image.open('test1.jpg')
    im1numpy = np.array(im1)
    im1numpy = im1numpy.flatten()
    im1numpy = np.reshape(im1numpy, [45045, 1])
    im2 = Image.open('test2.jpg')
    im2numpy = np.array(im2)
    im2numpy = im2numpy.flatten()
    im2numpy = np.reshape(im2numpy, [45045, 1])
    im3 = Image.open('test3.jpg')
    im3numpy = np.array(im3)
    im3numpy = im3numpy.flatten()
    im3numpy = np.reshape(im3numpy, [45045, 1])
    im4 = Image.open('test4.jpg')
    im4numpy = np.array(im4)
    im4numpy = im4numpy.flatten()
    im4numpy = np.reshape(im4numpy, [45045, 1])
    im5 = Image.open('test5.jpg')
    im5numpy = np.array(im5)
    im5numpy = im5numpy.flatten()
    im5numpy = np.reshape(im5numpy, [45045, 1])
    testing = [im1numpy,im2numpy,im3numpy,im4numpy,im5numpy]
    return testing
testing = testingVector()


corruptionLevel = 0.3

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            r'C:\Users\shrey\PycharmProjects\neural\savingModel\model',
        )
        inputForSaved = testing[0]
        maskForSaved = np.random.binomial(1, 1 - corruptionLevel, testing[0].shape)
        for op in graph.get_operations():
            print(str(op.name))
        print(tf.all_variables())
        #operation = graph.get_tensor_by_name("w:0")
        #out = sess.run(operation,feed_dict={inp:inputForSaved,masking:maskForSaved})
        print(out)










