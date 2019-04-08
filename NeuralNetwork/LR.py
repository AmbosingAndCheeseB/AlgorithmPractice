import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None], name = 'x')
Y = tf.placeholder(tf.float32, shape=[None], name = 'y')

W = tf.Variable(tf.random_normal([1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = W * X + b
loss = tf.reduce_mean(tf.square(hypothesis - Y))
tf.summary.scalar('loss', loss)


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
tensorboard_writer = tf.summary.FileWriter('./tensorboard_log', sess.graph)


for step in range(2019):
    loss_val, W_val, b_val, _ = \
    sess.run([loss, W, b, train],
                feed_dict = { X : [ 1, 2, 3, 4],
                     Y : [2.2 , 3.3, 4.4, 5.5]})

    summary = sess.run(merged, feed_dict = { X : [ 1, 2, 3, 4],
                                                    Y : [2.2 , 3.3, 4.4, 5.5]})

    tensorboard_writer.add_summary(summary, step)

    if step % 20 == 0:
        print(step, loss_val, W_val, b_val)
