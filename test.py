# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import glob
import parse_data
from tensorflow.contrib import learn


# Data Parameters
tf.flags.DEFINE_string("data_file", "./data/test.json", "Data source for the test data.")
tf.flags.DEFINE_boolean("with_labels", True, "Data with label?")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

if not FLAGS.checkpoint_dir:
    all_checkpoints = glob.glob('./runs/*')
    if len(all_checkpoints) > 0:
        FLAGS.checkpoint_dir = "%s/checkpoints" % (all_checkpoints[-1])
        print ("checkpoint_dir: %s" % FLAGS.checkpoint_dir)
    else:
        raise Exception('No checkpoint! Please run train.py first.')

print ("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print ("{}={}".format(attr.upper(), value._value))


x_text, y_test = parse_data.load_data_and_labels(FLAGS.data_file)
y_test = np.argmax(y_test, axis=1)
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_text)))


checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print (checkpoint_file)
print ("\nTesting...\n")
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = parse_data.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


# Print accuracy if y_test is defined
correct_predictions = float(sum(all_predictions == y_test))
print ("Total number of test examples: {}".format(len(y_test)))
print ("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
