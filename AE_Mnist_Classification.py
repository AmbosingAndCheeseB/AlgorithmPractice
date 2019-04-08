import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate_RMProp = 0.02
learning_rate_GradientDescent = 0.5
num_epochs = 100
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 128
hidden2_size = 64

x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, 10])

def build_AE(x):
    W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    out_h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    out_h2 = tf.nn.sigmoid(tf.matmul(out_h1, W2) + b2)
    W3 = tf.Variable(tf.random_normal(shape=[hidden2_size, hidden1_size]))
    b3 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    out_h3 = tf.nn.sigmoid(tf.matmul(out_h2, W3) + b3)
    W4 = tf.Variable(tf.random_normal(shape=[hidden1_size, input_size]))
    b4 = tf.Variable(tf.random_normal(shape=[input_size]))
    re_x = tf.nn.sigmoid(tf.matmul(out_h3, W4) + b4)

    return re_x, out_h2

def build_softmax_classifier(x):
    W_softmax = tf.Variable(tf.zeros([hidden2_size, 10]))
    b_softmax = tf.Variable(tf.zeros([10]))
    y_pred = tf.nn.softmax(tf.matmul(x,W_softmax) + b_softmax)

    return y_pred

y_pred, features = build_AE(x)
y_true = x
y_pred_softmax = build_softmax_classifier(features)

pretraining_loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
pretraining_train = tf.train.RMSPropOptimizer(learning_rate_RMProp).minimize(pretraining_loss)

finetuning_loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred_softmax), reduction_indices=[1]))
finetuning_train = tf.train.GradientDescentOptimizer(learning_rate_GradientDescent).minimize(finetuning_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples/batch_size)

    for epoch in range(num_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, pretraining_loss_print = sess.run([pretraining_train,pretraining_loss], feed_dict= {x: batch_xs})

        if epoch % display_step == 0:
            print("epoch : %d, loss : %f" % ((epoch+1, pretraining_loss_print)))

    for epoch in range(num_epochs+100):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, finetuning_loss_print = sess.run([finetuning_train, finetuning_loss], feed_dict= {x: batch_xs, y: batch_ys})

        if epoch % display_step == 0:
            print("epoch : %d, loss : %f" % ((epoch + 1, finetuning_loss_print)))

    correct_predition = tf.equal(tf.argmax(y,1), tf.argmax(y_pred_softmax, 1))
    accracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
    print("정확도 : %f" % sess.run(accracy, feed_dict={x:mnist.test.images, y: mnist.test.labels}))
