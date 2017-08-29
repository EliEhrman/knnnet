"""
The purpose of this module is to provide a test for the knnnet code

This simple module just creates the sequence/broken sequence data set and a 2-layer nn to provide
the first stage for the knnnet layer


"""

from __future__ import print_function
# import csv
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sys, getopt
import random
# import os
# import math
import collections

import v1


num_vals = 4
num_samples = 1000
core_dim = 8
num_steps = 1000
learn_rate = 4.0
acc_thresh = 0.8

c_mem_size = 1100
c_key_size = 4
c_num_classes = 2
c_num_qs = 50
c_num_ks = 3

flg_debug = False

try:
	opts, args = getopt.getopt(sys.argv[1:], "d")
except getopt.GetoptError:
	print('cknn_main.py [-d]')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-d':
		flg_debug = True

# function created to insert stops in the debugger. Create a stop/breakpoint by inserting a line like
# 				sess.run(t_for_stop)
def stop_reached(datum, tensor):
	if datum.node_name == 't_for_stop': # and tensor > 100:
		return True
	return False


t_for_stop = tf.constant(5.0, name='t_for_stop')

# np.random.seed(1)

data = np.ndarray([num_samples, num_vals], dtype=np.float32)
# truth = np.ndarray([num_samples, 2], dtype=np.float32)
truth = np.ndarray([num_samples], dtype=np.int32)
for isamp in range(num_samples / 2):
	epsilon = random.uniform(0., 0.01)
	end = random.uniform(epsilon * 2.0, 1.0)
	start = random.uniform(0.0, end - epsilon)
	if random.uniform(0., 1.0) > 0.5:
		data[isamp * 2, :] = np.linspace(start, end, num_vals)
	else:
		data[isamp * 2, :] = np.linspace(end, start, num_vals)
	# There are two alternatives here. One creates a simple random sequence
	# in a similar range but with no likelihood of being a sequence
	# The other, creates a sequence but messes up a single element of the sequence
	# v1:
	data[isamp*2+1, :]=np.random.uniform(start, end, num_vals)
	# v2
	# data[isamp*2+1, :]=data[isamp*2, :]
	# data[isamp * 2 + 1, random.randint(0, num_vals-1)] = random.uniform(start, end)
	# end of versions
	truth[isamp * 2] = 1
	truth[isamp * 2 + 1] = 0

last_train = num_samples * 9 / 10

t_data = tf.constant(data, name='t_data')
t_truth = tf.constant(truth, name='t_truth')
t_index = tf.Variable(tf.random_uniform([c_num_qs], 0, num_samples - 1, dtype=tf.int32), trainable=False, name='t_index')

with tf.name_scope('nn_params'):
	wfc1 = tf.Variable(tf.random_normal([num_vals, core_dim], 0, 1e-3, dtype=tf.float32), name='t_wfc1')
	bfc1 = tf.Variable(tf.random_normal([core_dim], 0, 1e-5, dtype=tf.float32), name='t_bfc1')
	wfc2 = tf.Variable(tf.random_normal([core_dim, c_key_size], 0, 1e-3, dtype=tf.float32), name='t_wfc2')
	bfc2 = tf.Variable(tf.random_normal([c_key_size], 0, 1e-5, dtype=tf.float32), name='t_bfc2')

with tf.name_scope('data_setup'):
	t_index_set_op = tf.assign(t_index, tf.random_uniform([c_num_qs], 0, last_train - 1, dtype=tf.int32), name='t_index_set_op')
	t_index_set_op_test = tf.assign(t_index,
									tf.random_uniform([c_num_qs], last_train, num_samples - 1, dtype=tf.int32),
									name='t_index_set_op_test')
	t_index_set_op_dummy = tf.assign(t_index, tf.range(c_num_qs, dtype=tf.int32), name='t_index_set_op_dummy')
	t_batch = tf.gather(t_data, t_index, name='t_batch')
	t_batch_truth = tf.gather(t_truth, t_index, name='t_batch_truth')

with tf.name_scope('nn'):
	fc1 = tf.sigmoid(tf.matmul(t_batch, wfc1, name='matmul_fc1') + bfc1, name='t_fc1')
	pred = tf.sigmoid(tf.matmul(fc1, wfc2, name='matmul_fc2') + bfc2, name='t_pred')

"""
with tf.name_scope('evaluation'):
	t_err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
																   labels=t_batch_truth), name='x_entropy')
	train1 = tf.train.GradientDescentOptimizer(learning_rate=learn_rate, name='GDO').minimize(t_err, name='GDOmin')
	tf.summary.scalar('Error', t_err)

	t_correct = tf.equal(tf.argmax(pred, 1, name='am_pred'), tf.argmax(t_batch_truth, 1, name='am_truth'), name='eq_correct')
	t_acc = tf.reduce_mean(tf.cast(t_correct, tf.float32), name='acc_mean')
	tf.summary.scalar('Accuracy', t_acc)
"""

op_mem_init, op_loss_constants_set, t_loss, ph_q_labels, ph_update_q_num, op_mem_updates, op_q_calcs, t_acc\
	= v1.init_knnnet(c_mem_size, c_key_size, c_num_qs, c_num_classes, c_num_ks, t_q_keys_unnorm=pred, t_q_labels=t_batch_truth)

train1 = tf.train.GradientDescentOptimizer(learning_rate=learn_rate, name='GDO').minimize(t_loss, name='train1')

sess = tf.Session()
if flg_debug:
	sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
	sess.add_tensor_filter("stop_reached", stop_reached)
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
									 sess.graph)
sess.run(tf.global_variables_initializer())

r_batch, r_wfc1, r_bfc1, r_fc1 = sess.run([t_batch, wfc1, bfc1, fc1])

sess.run(op_mem_init)

for step in range(num_steps + 1):
	sess.run(t_index_set_op)
	sess.run(op_loss_constants_set)
	# sess.run(t_for_stop)
	# Calculate the loss which start a calculation from the level of batch
	r_loss = sess.run(t_loss)
	print('step:', step, 'acc:', sess.run(t_acc))
	# set some of the values in stone which we need for the memory modification
	sess.run(op_q_calcs)
	# train all network parameters including calculating a loss function facing backwards from the
	sess.run(train1)
	for iq in range(c_num_qs):
		sess.run(op_mem_updates, feed_dict={ph_update_q_num:iq})


sess.close()

print('done')