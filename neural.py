import sys
import math
from PIL import Image
import numpy as np
import tensorflow as tf


'''im1 = Image.open('1.jpg')
im1numpy = np.array(im1)
print(im1numpy.shape,"before")
im1numpy = im1numpy.flatten()
print(im1numpy.shape)
print(im1numpy)'''

def trainingVector():
    im1 = Image.open('1.jpg')
    im1numpy = np.array(im1)
    im1numpy = im1numpy.flatten()
    im1numpy=np.reshape(im1numpy,[45045,1])
    im2 = Image.open('2.jpg')
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
    im6 = Image.open('6.jpg')
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

training = trainingVector()

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
testing = np.array(testing)

width = 195
height = 231
inputNodes = 195*231
hiddenNodes = 27027
corruptionLevel = 0.3


inp = tf.placeholder("float", [inputNodes,1],name='inp')


masking = tf.placeholder("float",[inputNodes,1],name='masking')

weight_initialMax = 4*np.sqrt(6./(inputNodes+hiddenNodes))
weight_initial = tf.random_uniform(shape=[inputNodes,hiddenNodes],minval=weight_initialMax,maxval=weight_initialMax)


weight = tf.Variable(weight_initial,name='weight')
bias = tf.Variable(tf.zeros([1,hiddenNodes]),name='bias')

primeWeight = tf.transpose(weight)
primeBias = tf.Variable(tf.zeros([inputNodes]),name='primeBias')

def model(inp,masking,weight,bias,primeWeight,primeBias):
    corruptinginp = masking*inp
    corruptinginp = tf.transpose(corruptinginp)

    hidden = tf.nn.sigmoid(tf.matmul(corruptinginp,weight) + bias)
    reconstruction = tf.nn.sigmoid(tf.matmul(hidden,primeWeight) + primeBias)
    return reconstruction

reconstruction = model(inp,masking,weight,bias,primeWeight,primeBias)

cost = tf.reduce_sum(tf.pow(inp-reconstruction,2))
trainingOptimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

def hiddenOutput(inp,masking,weight,bias):
    corrupting = masking*inp
    corrupting = tf.transpose(corrupting)
    out = tf.nn.sigmoid(tf.matmul(corrupting,weight)+bias)
    return out
out = hiddenOutput(inp,masking,weight,bias)

inputForSaved = tf.placeholder(shape=[],dtype=tf.float32)
maskingForSaved = tf.placeholder(shape=[],dtype=tf.float32)
outputForSaved = tf.placeholder(shape=[],dtype=tf.float32,name="outputForSaved")
outputForSaved = out


trainImagesLabel = [1,2,3,4,5,6,7]
testImagesLabel = [1,4,6,7,7]
count = 0

def saving(sess):
    with tf.Graph().as_default():

            # Saving
        inputs = {
            "inputForSaved":inputForSaved,
            "maskingForSaved": maskingForSaved,
        }
        outputs = {"outputForSaved": out}
        tf.saved_model.simple_save(
            sess, r'C:\Users\shrey\PycharmProjects\neural\savingModel\model', inputs, outputs
        )

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    for i in range(1):
        for j in range(3):
            input_ = training[j]
            mask_np = np.random.binomial(1,1-corruptionLevel,input_.shape)
            sess.run(trainingOptimizer,feed_dict={inp:input_,masking:mask_np})
            count = count+1
            print(count," iteration")

    saving(sess)

