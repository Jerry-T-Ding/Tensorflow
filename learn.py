#coding:utf-8
#0导入模块。

import numpy as np
import tensorflow as tf
import data

X = data.data[['D/G','L/G']]
X = X.values.reshape(-1,2)
#print (X)
Y = data.data[['LnGDP']]
Y = Y.values.reshape(-1,1)
#print (Y)
n = X.shape[0]
#print (n)
print ("\n")

#1定义神经网络的输入、参数和输出,定义前向传播过程。

x = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))
w = tf.Variable(tf.random_normal([2, 1],name="weight"))
b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)
y_ = tf.add(tf.matmul(x, w), b)

#2定义损失函数及反向传播方法。
#loss = tf.reduce_mean(tf.square(y-y_))
loss = tf.reduce_sum(tf.pow(y-y_, 2)) / (2 * n)
#train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.012).minimize(loss)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 训练模型。
    STEPS = 50001
    for i in range(STEPS):
        start = i % 18
        end = start + 1
        sess.run(train_step, feed_dict={x: X[start:end], y: Y[start:end]})
        if i % 1000 == 0 :
            total_loss = sess.run(loss, feed_dict={x: X, y: Y})
            print ("After %d training step(s), loss on all data is %g" % (i, total_loss))
            print ("w:\n", sess.run(w))
            print ("b:\n", sess.run(b))
            print ("\n")

