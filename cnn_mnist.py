from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # [batch_size, 28, 28, 1]
  # 1st convolutional layer
  conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) # [batch_size, 28, 28, 32]
  # 1st pooling layer
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2) # [batch_size, 14, 14, 32]
  # 2nd convolutional layer
  conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) # [batch_size, 14, 14, 64]
  # 2nd pooling layer
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2) # [batch_size, 7, 7, 64]
  # Flattening tensor to two dimensions
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # [batch_size, 3136]
  # Dense layer
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu) # [batch_size, 1024]
  # Dropout regularization
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN) # [batch_size, 1024]
  # Logits layer
  logits = tf.layers.dense(inputs=dropout, units=10) # [batch_size, 10]
  # Predictions
  predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  # Calculate loss
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
  # Training optimizer
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  # Evaluation metrics
  eval_metrics_ops = {
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metrics_ops=eval_metrics_ops)

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  # Create estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
  # Logging hook
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
  # Train model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True
  )
  mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
  # Evaluate model
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False
  )
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
