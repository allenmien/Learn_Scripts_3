# -*-coding:utf-8-*-
"""
@Time   : 2019-02-14 14:53
@Author : Mark
@File   : exercise.py
"""
import tensorflow as tf

x = tf.get_variable(
    name='x', shape=[1], initializer=tf.truncated_normal_initializer())
loss = tf.pow(x=(x - 2), y=2, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate=1)
grad_and_var = optimizer.compute_gradients(
    loss=loss, var_list=tf.trainable_variables())
grad_and_var_list = list()
for g, v in grad_and_var:
    g = tf.clip_by_value(g, clip_value_min=-1, clip_value_max=1)
    grad_and_var_list.append([g, v])
train_step = optimizer.apply_gradients(tuple(grad_and_var_list))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10):
        train_step.run()
        print('x : ', sess.run(x), '\nloss : ', sess.run(loss))
        print(sess.run(grad_and_var))
