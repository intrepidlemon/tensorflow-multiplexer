"""Builds a multilayer perceptron

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

Much inspiration from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
import argparse
import os
import sys
import time
import math

import tensorflow as tf

import input_data

# Basic model parameters as external flags.
FLAGS = None


def inference(features, hidden1_units, hidden2_units):
    """Build the MLP model up to where it may be used for inference.
    Args:
      features: Features placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.
    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal(
                [FLAGS.bits, hidden1_units],
                stddev=1.0 / math.sqrt(float(FLAGS.bits))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(features, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden1_units, hidden2_units],
                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden2_units, 2],
                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([2]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss_func(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
      logits: Logits tensor, float - [batch_size, n_classes].
      labels: Labels tensor, int32 - [batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, n_classes].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, n_classes).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    features_placeholder: Features placeholder.
    labels_placeholder: Labels placeholder.
  """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    features_placeholder = tf.placeholder(
        tf.float32, shape=(batch_size, FLAGS.bits))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return features_placeholder, labels_placeholder

def create_fill_feed_dict(data_set, features_pl, labels_pl):
    data_set = data_set.batch(FLAGS.batch_size)
    iterator = data_set.make_initializable_iterator()
    def fill_feed_dict(sess):
        """Fills the feed_dict for training the given step.
        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }
        Args:
          data_set: The set of features and labels, from input_data.read_data_sets()
          features_pl: The features placeholder, from placeholder_inputs().
          labels_pl: The labels placeholder, from placeholder_inputs().
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next
        # `batch size` examples.
        features_feed, labels_feed = iterator.get_next()
        features, labels = sess.run([features_feed, labels_feed])
        feed_dict = {
            features_pl: features,
            labels_pl: labels,
        }
        return feed_dict
    return fill_feed_dict, iterator.initializer


def do_eval(sess, eval_correct, features_placeholder, labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      features_placeholder: The features placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of features and labels to evaluate, from
        input_data
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = FLAGS.N // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    fill_feed_dict, initializer = create_fill_feed_dict(data_set, features_placeholder, labels_placeholder)
    sess.run(initializer)
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(sess)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of features and labels for training, validation, and
    # test on MNIST.
    data_sets = input_data.load_multiplexer(FLAGS.input_data_dir, FLAGS.bits)

    # Generate placeholders for the features and labels.
    features_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(features_placeholder, FLAGS.hidden1, FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = loss_func(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    fill_feed_dict, initializer = create_fill_feed_dict(data_sets.train, features_placeholder, labels_placeholder)

    sess.run(initializer)

    # Start the training loop.
    for step in range(FLAGS.max_steps):
        start_time = time.time()

        # Fill a feed dictionary with the actual set of features and labels
        # for this particular training step.
        feed_dict = fill_feed_dict(sess)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                       duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)
            # Evaluate against the training set.
            print('Training Data Eval:')
            do_eval(sess, eval_correct, features_placeholder,
                    labels_placeholder, data_sets.train)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(sess, eval_correct, features_placeholder,
                    labels_placeholder, data_sets.test)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bits',
        type=int,
        default=6,
        help='Number of bits of multiplexer problem')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10,
        help='Number of steps to run trainer.')
    parser.add_argument(
        '--N',
        type=int,
        default=500,
        help='Number of entries in the dataset')
    parser.add_argument(
        '--hidden1',
        type=int,
        default=5,
        help='Number of units in hidden layer 1.')
    parser.add_argument(
        '--hidden2',
        type=int,
        default=4,
        help='Number of units in hidden layer 2.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size.  Must divide evenly into the dataset sizes.')
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default="datasets",
        help='Directory to put the input data.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default="logs",
        help='Directory to put the log data.')
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
