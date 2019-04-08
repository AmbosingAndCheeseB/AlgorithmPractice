import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))

logits = tf.matmul(x, W) + b
y_pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices=[1]))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
tf.summary.scalar('loss', loss)


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
tensorboard_writer = tf.summary.FileWriter('./tensorboard_log', sess.graph)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    summary = sess.run(merged, feed_dict={x: batch_xs,
                                          y: batch_ys})

    tensorboard_writer.add_summary(summary, i)


correction_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
print("정확도 : %f" % sess.run(accuracy, feed_dict= {x:mnist.test.images, y : mnist.test.labels}))

sess.close()