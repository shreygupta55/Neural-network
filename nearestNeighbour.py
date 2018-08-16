import numpy as np
import tensorflow as tf
from PIL import  Image

def trainingVector():
    im1 = Image.open('2.jpg')
    im1numpy = np.array(im1)
    im1numpy = im1numpy.flatten()
    im1numpy=np.reshape(im1numpy,[45045,1])
    im2 = Image.open('6.jpg')
    im2numpy = np.array(im2)
    im2numpy = im2numpy.flatten()
    im2numpy = np.reshape(im2numpy, [45045, 1])
    im3 = Image.open('3.jpg')
    im3numpy = np.array(im3)
    im3numpy = im3numpy.flatten()
    im3numpy = np.reshape(im3numpy, [45045, 1])
    im4 = Image.open('4.jpg')
    im4numpy = np.array(im4)
    im4numpy = im4numpy.flatten()
    im4numpy = np.reshape(im4numpy, [45045, 1])
    im5 = Image.open('5.jpg')
    im5numpy = np.array(im5)
    im5numpy = im5numpy.flatten()
    im5numpy = np.reshape(im5numpy, [45045, 1])
    im6 = Image.open('1.jpg')
    im6numpy = np.array(im6)
    im6numpy = im6numpy.flatten()
    im6numpy = np.reshape(im6numpy, [45045, 1])
    im7 = Image.open('7.jpg')
    im7numpy = np.array(im7)
    im7numpy = im7numpy.flatten()
    im7numpy = np.reshape(im7numpy, [45045, 1])
    im8 = Image.open('8.jpg')
    im8numpy = np.array(im8)
    im8numpy = im8numpy.flatten()
    im8numpy = np.reshape(im8numpy, [45045, 1])
    training = [im1numpy,im2numpy,im3numpy,im4numpy,im5numpy,im6numpy,im7numpy,im8numpy]
    return training


trainingLabels = [1,2,3,4,5,6,7,8]
trainingLabels = np.array(trainingLabels)

tr = tf.placeholder("float", [None,45045,1],name='tr')
te = tf.placeholder("float",[45045,None],name='te')

distance = tf.reduce_sum(tf.abs(tf.add(tr,tf.negative(te))),reduction_indices=1)

#pred = tf.argmin(distance,0)
pred = tf.argmin(distance)
print(pred)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training = trainingVector()
    im = Image.open('test2.jpg')
    imtest = np.array(im, dtype=float)
    imtest = imtest.flatten()
    imtest = np.reshape(imtest,[45045, 1])
    itest = np.zeros([45045, 1],dtype = float)
    print(imtest)
    nn_index = sess.run(pred, feed_dict={tr: training, te: imtest})
    print("Prediction:",(trainingLabels[nn_index]))

