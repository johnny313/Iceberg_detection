"""A deep classifier using convolutional layers.
uses the STATOIL Icebarge SAR dataset to discriminate between ships and icebergs
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import json
import math
import numpy as np
import tensorflow as tf

FLAGS = None
IMAGE_SIZE = 75
KERNEL_SIZE = 5
NUM_CLASSES = 2
NUM_FEAT_CONVO_1 = 32
NUM_FEAT_CONVO_2 = 64


def deepnn(x, conv1_features, conv2_features, fc1_units, num_classes):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 5625), where 5625 is the
    number of pixels in an image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 2), with values
    equal to the logits of classifying the image as iceberg or ship. keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 2])

  # First convolutional layer - maps each 2 band image to conv1_features feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, 2, conv1_features])
    b_conv1 = bias_variable([conv1_features])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
    activation_summaries(h_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')

  # Second convolutional layer -- maps conv1_features to conv2_feature feature maps.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, conv1_features, conv2_features])
    b_conv2 = bias_variable([conv2_features])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
    activation_summaries(h_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')

  #create fully connected layer, mapping each convo feature to a new feature in a flat structure 
  with tf.name_scope('fc1'):
    #fc1_units here should be 
    #1) the number of features from the convo/pooling chain 
    #2) the number of features genreated in the fully connected layer

    W_fc1 = tf.Variable(
                    tf.truncated_normal([fc1_units, fc1_units],
                    stddev=1.0 / math.sqrt(float(fc1_units))),
                    name='weights')
    variable_summaries(W_fc1)
    b_fc1 = biases = tf.Variable(tf.zeros([fc1_units]),
                                         name='biases')
    variable_summaries(b_fc1)

    h_pool2_flat = tf.reshape(h_pool2, [-1, fc1_units])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([fc1_units, num_classes])
    variable_summaries(W_fc2)
    b_fc2 = bias_variable([num_classes])
    variable_summaries(b_fc2)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1) #initializer=tf.contrib.layers.xavier_initializer_conv2d()
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)#initializer=tf.constant_initializer(0.05)
  return tf.Variable(initial)

def generate_batch(data, batch_size):
  ii = np.random.choice(len(data[0]), batch_size, replace=False)
  return (np.array([data[0][i] for i in ii]), np.array([data[1][i] for i in ii]))

def load_data(path, test_fraction):
  # functions for processing the json-format study data 
  def get_label(dat_dict):
    lab = np.array([0,0])
    lab[dat_dict['is_iceberg']] = 1.
    return lab

  def get_image(dat_dict):
    return np.array([dat_dict['band_1'],dat_dict['band_2']]).T

  with open(path, 'r+b') as fle:
    dat = json.load(fle)
    #put the images and labels into an array where each row is one observation
    dataset = np.array([[get_image(d) for d in dat], [get_label(d) for d in dat]]).T
  
  #break into test / train data
  np.random.shuffle(dataset)
  test_bool = np.zeros(dataset.shape[0])
  num_test = int(dataset.shape[0] * test_fraction)
  test_bool[np.random.choice(dataset.shape[0], num_test, replace=False)] = 1
  test = dataset[np.where(test_bool==1)]
  train = dataset[np.where(test_bool==0)]
  return (test[:,0], test[:,1]), (train[:,0], train[:,1])

def variable_summaries(var):
  #attach summaries for tensorboard to read, visualize
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram',var)

def activation_summaries(var):
  tf.summary.histogram('histogram', var)
  tf.summary.scalar('sparsity', tf.nn.zero_fraction(var))

def main(_):
  # Import data
  test, train = load_data(FLAGS.data_dir, 0.2)

  # Create the model
  x = tf.placeholder(tf.float32, [None, IMAGE_SIZE*IMAGE_SIZE,2])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  # Build the graph for the deep net
  #get the size of the fully connected layer based on the input image size, the kernel size, and the number of convo features
  size_after_conv_and_pool_twice = int(math.ceil((math.ceil(float(IMAGE_SIZE-KERNEL_SIZE+1)/2)-KERNEL_SIZE+1)/2))
  num_fully_connected_features = size_after_conv_and_pool_twice**2 * NUM_FEAT_CONVO_2 
  y_conv, keep_prob = deepnn(x, NUM_FEAT_CONVO_1, NUM_FEAT_CONVO_2, num_fully_connected_features, NUM_CLASSES)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy',cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  tf.summary.scalar('accuracy',accuracy)


  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(graph_location+'/train')
  train_writer.add_graph(tf.get_default_graph())
  test_writer = tf.summary.FileWriter(graph_location+'/test')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #initialize saver to restore model later
    saver = tf.train.Saver()
    test_batch = generate_batch(test, test[0].shape[0])
    for i in range(2000):
      batch = generate_batch(train, 50)
      summary_train, _ = sess.run([merged, train_step],feed_dict={x: batch[0], 
                                                                  y_: batch[1], 
                                                                  keep_prob: 0.5})
      if i % 100 == 0:
        summary_test, acc = sess.run([merged, accuracy], feed_dict={x: test_batch[0], 
                                                                    y_: test_batch[1], 
                                                                    keep_prob: 1.0})
        test_writer.add_summary(summary_test, i)
        train_writer.add_summary(summary_train, i)
        print('step %d, test accuracy %g' % (i, acc))
      #train_step.run(feed_dict={x: batch[0], 
      #                          y_: batch[1], 
      #                          keep_prob: 0.5})
    
    print('final test accuracy %g' % accuracy.eval(feed_dict={
        x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

    # Save model weights to disk
    save_path = saver.save(sess, FLAGS.model_dir)
    print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/home/real_john/tensorflow/train_data/processed/train.json',
                      help='Directory for storing input data')
  parser.add_argument('--model_dir', type=str,
                      default='/home/real_john/tensorflow/model.ckpt',
                      help='Directory for saving the model')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


