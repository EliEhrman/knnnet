import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import sys, getopt


def t_repeat(t, rep_axis, tile_vec, name=None):
	return  tf.tile(tf.expand_dims(t, rep_axis), tile_vec, name=name)

# converts an array of numbers into a selection index for a a 2d tensor (matrix) that selects one from each row
def t_sel_index(t, len, name=None):
	return tf.stack([tf.range(len), t], axis=1, name=name)



# tr = tf.random_uniform([11, 4], 0, 100, dtype=tf.int32, name='tr')
# # ti = tf.stack([tf.range(11), tf.random_uniform([11], 0, 4, dtype=tf.int32)], axis=1, name='ti')
# ti = t_sel_index(tf.random_uniform([11], 0, 4, dtype=tf.int32), 11, name='ti')
# t1 = tf.gather_nd(tr, ti, name='t1')

c_max_init_age = 1000


def init_knnnet(mem_size, key_size, num_qs, num_classes, num_ks, t_q_keys_unnorm, t_q_labels):
	with tf.name_scope('mem_init'):

		nd_mem_keys = np.random.normal(loc=0.0, scale=0.1, size=[mem_size, key_size])
		nd_mem_labels = np.random.randint(low=0, high=num_classes, size=[mem_size], dtype=np.int32)

		v_mem_keys = tf.Variable(tf.zeros([mem_size, key_size], dtype=tf.float32), name='v_mem_keys')
		op_mem_keys_init = tf.assign(	v_mem_keys,
										tf.nn.l2_normalize(tf.constant(nd_mem_keys, dtype=tf.float32), dim=1),
									 name='op_mem_keys_init')
		v_mem_labels = tf.Variable(tf.zeros([mem_size], dtype=tf.int32), name='v_mem_labels')
		op_mem_labels_init = tf.assign(	v_mem_labels, tf.constant(nd_mem_labels, dtype=tf.int32),
										name='op_mem_labels_init')

		v_mem_ages = tf.Variable(tf.zeros([mem_size], dtype=tf.int32), name='v_mem_ages')
		op_mem_ages_init = tf.assign(v_mem_ages,
									tf.random_uniform([mem_size], minval=c_max_init_age / 2,
													  maxval=c_max_init_age, dtype=tf.int32),
									name='op_mem_ages_init')

		op_mem_init = [op_mem_keys_init, op_mem_labels_init, op_mem_ages_init]

		# t_q_keys = tf.nn.l2_normalize(tf.random_normal(shape=[num_qs, key_size], mean=0.0, stddev=0.2, dtype=tf.float32), dim=1, name='t_q_keys')
		t_q_keys = tf.nn.l2_normalize(t_q_keys_unnorm, dim=1, name='t_q_keys')
		# t_q_keys = tf.nn.l2_normalize(tf.random_normal(shape=[num_qs, key_size], mean=0.0, stddev=0.2, dtype=tf.float32), dim=1, name='t_q_keys')
		# t_q_labels = tf.random_uniform(shape=[num_qs], minval=0, maxval=num_classes, dtype=tf.int32, name='t_q_labels')
		# t_q_keys = tf.placeholder(dtype=tf.float32, shape=[num_qs, key_size], name='t_q_keys')
		#t_q_labels = tf.placeholder(shape=[num_qs], dtype=tf.int32, name='t_q_labels')

	with tf.name_scope('match_and_mismatch'):
		# For the purposes of the following rows a mismatch is an entry whether in the database or the top k of the db where the label is not the same as that of the training query
		t_q_label_match = tf.equal(t_repeat(t_q_labels, [-1], [1, mem_size]), t_repeat(v_mem_labels, [0], [num_qs, 1]), name='t_q_label_match')
		t_q_label_at_least_one_match = tf.reduce_any(t_q_label_match, axis=1, name='t_q_label_at_least_one_match')
		t_q_label_all_matched = tf.reduce_all(t_q_label_match, axis=1, name='t_q_label_all_matched')
		t_mem_range = t_repeat(tf.range(mem_size), [0], [num_qs, 1], name='t_mem_range')
		t_mem_rand = tf.random_uniform(shape=[num_qs, mem_size], minval=0, maxval=mem_size, dtype=tf.int32, name='t_mem_rand')
		t_mem_notsel = tf.fill(dims=[num_qs, mem_size], value=mem_size, name='t_mem_notsel')
		t_mem_rand_for_match = tf.where(t_q_label_match, t_mem_rand, t_mem_notsel, name='t_mem_rand_for_match')
		t_mem_rand_for_mismatch = tf.where(t_q_label_match, t_mem_notsel, t_mem_rand, name='t_mem_rand_for_mismatch')
		t_mem_match_idx = tf.cast(tf.arg_min(t_mem_rand_for_match, dimension=1), dtype=tf.int32, name='t_mem_match_idx')
		t_mem_mismatch_idx = tf.cast(tf.arg_min(t_mem_rand_for_match, dimension=1), dtype=tf.int32, name='t_mem_mismatch_idx')
		t_mem_match_key = tf.gather(v_mem_keys, tf.where(t_q_label_at_least_one_match, t_mem_match_idx, tf.zeros([num_qs], dtype=tf.int32)), name='t_mem_match_key')
		t_mem_mismatch_key = tf.gather(v_mem_keys, tf.where(t_q_label_all_matched, tf.zeros([num_qs], dtype=tf.int32), t_mem_mismatch_idx), name='t_mem_mismatch_key')


		t_running_idx = t_repeat(tf.range(num_ks), [0], [num_qs, 1], name='t_running_idx')
		t_above_range = tf.fill(dims=[num_qs, num_ks], value=mem_size, name='t_above_range')
		t_cds = tf.transpose(tf.matmul(v_mem_keys, t_q_keys, transpose_b=True), name='t_cds')
		t_top, t_top_idxs = tf.nn.top_k(t_cds, num_ks, name='t_top_idxs')
		t_n1 = t_top_idxs[:, 0]
		t_k1 = tf.gather(v_mem_keys, t_n1, name='t_k1')
		t_hit_or_not = tf.equal(t_q_labels, tf.gather(v_mem_labels, t_n1), name='t_hit_or_not')
		t_acc = tf.divide(tf.reduce_sum(tf.cast(t_hit_or_not, dtype=tf.float32)), tf.cast(num_qs, dtype=tf.float32), name='t_acc')

		t_top_q_labels = tf.gather(v_mem_labels, t_top_idxs, name='t_top_q_labels')
		t_top_q_match = tf.equal(t_top_q_labels, t_repeat(t_q_labels, [-1], [1, num_ks]), name='t_top_q_match')
		t_top_q_match_any = tf.reduce_any(t_top_q_match, axis=1, name='t_top_q_match_any')
		t_top_q_match_all = tf.reduce_all(t_top_q_match, axis=1, name='t_top_q_match_all')

		# note. Not the db index returned here but the index of the sorted top k
		# we are just finding the lowest of the range (running index) that has not been converted to an above range value by the boolean tensor
		# shape = [num_qs]
		t_first_match_top_idx = tf.where(t_top_q_match, t_running_idx, t_above_range, name='t_first_match_top_idx')
		# same for mismatch. This time the bool tensor reverses the effect because the t_above_range is the param before the running index
		t_first_mismatch_top_idx = tf.where(t_top_q_match, t_above_range, t_running_idx, name='t_first_mismatch_top_idx')
		# index among the sorted top k of the first mismatch
		# all rows where there were no mismatches in the top k, are replaced by a zero
		# the value is an index into the top k and NOT and index into the DB
		# note 0 is a valid value for the index. This case therefore generates the WRONG value
		# but is replaced by the where on the key extractipn. Why apply the *where* twice? To avoid array overrun
		# shape=[num_qs] . i.e. one value for each query
		t_first_mismatch_range_idx =  tf.where(t_top_q_match_all,
											   tf.zeros([num_qs], dtype=tf.int32),
											   tf.reduce_min(t_first_mismatch_top_idx, axis=1),
											   name='t_first_mismatch_range_idx')
		# same for the first match from among the sorted top k
		# this time we replace the index of any query where there was NO match by a zero
		# note that the if else args were switched relatiive to the mismatch case
		# shape=[num_qs] . i.e. one value for each query
		t_first_match_range_idx =  tf.where(t_top_q_match_any,
											tf.reduce_min(t_first_match_top_idx, axis=1),
											tf.zeros([num_qs], dtype=tf.int32),
											name='t_first_match_range_idx')
		# convert the index found to an index into the db. THis is done by selecting which of the top k for each q is wanted and getting its value
		# shape=[num_qs] . i.e. one value for each query
		t_match_idx = tf.gather_nd(t_top_idxs, t_sel_index(t_first_match_range_idx, num_qs), name='t_match_idx')
		t_mismatch_idx = tf.gather_nd(t_top_idxs, t_sel_index(t_first_mismatch_range_idx, num_qs), name='t_mismatch_idx')
		# get the key. This is ALMOST the final value. It containes an error if there was no match (for mismatch key, no mismatch) in the top k
		# This uses the index just retrieved to retrieve the key itself from the mem db
		# shape=[num_qs, key_size]
		t_top_q_match_key = tf.gather(v_mem_keys, t_match_idx, name='t_top_q_match_key')
		t_top_q_mismatch_key = tf.gather(v_mem_keys, t_mismatch_idx, name='t_top_q_mismatch_key')
		# combine the top k match key with the top mem match key, inserting the latter if there was no match in the key
		# This is the key we want, called kp in Kaiser and Nachum's paper. It is the closest neighbor that matches the query (training) label
		# or, if no neighbor matches, a random key that matches, or if there is no label match in the db, a zero key. The zero key does not
		# matter because no loss function is created in that case
		# shape=[num_qs, key_size]
		t_match_key = tf.where(t_top_q_match_any, t_top_q_match_key, t_mem_match_key, name='t_match_key')
		# note reversal of arguments. Called kb in the paper
		t_mismatch_key = tf.where(t_top_q_match_all, t_mem_mismatch_key, t_top_q_mismatch_key, name='t_mismatch_key')

	with tf.name_scope('loss_calc'):
		v_loss_constants_1 = tf.Variable(tf.zeros([num_qs, key_size], dtype=tf.float32), name='v_loss_constants_1')
		v_loss_constants_2 = tf.Variable(tf.zeros([num_qs, key_size], dtype=tf.float32), name='v_loss_constants_2')
		op_loss_constants_1_set = tf.assign(v_loss_constants_1, t_mismatch_key, name='op_loss_constants_1_set')
		op_loss_constants_2_set = tf.assign(v_loss_constants_2, t_match_key, name='op_loss_constants_2_set')
		op_loss_constants_set = [op_loss_constants_1_set, op_loss_constants_2_set]

		t_loss_cds_1 = tf.transpose(tf.reduce_sum(tf.multiply(v_loss_constants_1, t_q_keys), axis=1), name='t_loss_cds_1')
		t_loss_cds_2 = tf.transpose(tf.reduce_sum(tf.multiply(v_loss_constants_2, t_q_keys), axis=1), name='t_loss_cds_2')
		alpha = 0.1
		# t_loss = tf.nn.relu(tf.subtract(tf.nn.relu(tf.add(tf.subtract(t_loss_cds_1, t_loss_cds_2), alpha)), alpha), name='t_loss')
		t_loss = tf.subtract(t_loss_cds_1, t_loss_cds_2, name='t_loss')

	with tf.name_scope('q_calc_freeze'):
		# The following ops set the input and calculated tensors in stone as variables
		# However, it is important not to do it too early as the loss neeeds to propagate all the way back to the initial batch
		# The good news is that, once set, you can do a learn that will change all the weights of previous layers
		v_n1_new_key = tf.Variable(tf.zeros([num_qs, key_size], dtype=tf.float32), name='v_n1_new_key')
		op_n1_new_key_set = tf.assign(v_n1_new_key, tf.divide(tf.add(t_k1, t_q_keys), 2.0), name='op_n1_new_key_set')
		v_hit_or_not = tf.Variable(tf.zeros([num_qs], dtype=tf.bool), name='v_hit_or_not')
		op_hit_or_not_set = tf.assign(v_hit_or_not, t_hit_or_not, name='op_hit_or_not_set')
		v_q_keys = tf.Variable(tf.zeros([num_qs, key_size], dtype=tf.float32), name='v_q_keys')
		op_q_keys_set = tf.assign(v_q_keys, t_q_keys, name='op_q_keys_set')
		v_n1 = tf.Variable(tf.zeros([num_qs], dtype=tf.int32), name='v_n1')
		op_n1_set= tf.assign(v_n1, t_n1, name='op_n1_set')

		op_q_calcs = [op_n1_new_key_set, op_hit_or_not_set, op_q_keys_set, op_n1_set]

	with tf.name_scope('mem_update'):
		ph_update_q_num = tf.placeholder(dtype=tf.int32, name='ph_update_q_num')
		t_oldest_idx = tf.cast(tf.argmax(v_mem_ages, axis=0), dtype=tf.int32, name='t_oldest_idx')
		t_replace_idx = tf.where(v_hit_or_not[ph_update_q_num], v_n1[ph_update_q_num], t_oldest_idx)
		t_new_key = t_repeat(	tf.where(	v_hit_or_not[ph_update_q_num],
											v_n1_new_key[ph_update_q_num],
											v_q_keys[ph_update_q_num]),
								[0], [mem_size, 1], name='t_new_key')
		t_replace_mrk = tf.equal(tf.range(mem_size), t_replace_idx, name='t_replace_mrk' )

		op_db_mem_keys_update = tf.assign(v_mem_keys, tf.where(t_replace_mrk, t_new_key, v_mem_keys), name='op_db_mem_keys_update')
		t_ages_incr = tf.add(v_mem_ages, 1)
		op_ages_update = tf.assign(	v_mem_ages,
									tf.where(t_replace_mrk, tf.zeros([mem_size], dtype=tf.int32), t_ages_incr),
									name='op_ages_update')
		op_mem_updates = [op_db_mem_keys_update, op_ages_update]

	return op_mem_init, op_loss_constants_set, t_loss, t_q_labels, ph_update_q_num, op_mem_updates, op_q_calcs, t_acc





