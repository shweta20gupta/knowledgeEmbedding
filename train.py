#! /usr/bin/env python
import glob
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from complex_embeddings import ComplexLogistic
from tensorflow.contrib import learn
import logging
import logging.config
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularizaion lambda (default: 0.0)")
 

#Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 20)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("max_iter", 1000, "")
tf.flags.DEFINE_float("neg_ratio", 2, "")
tf.flags.DEFINE_float("contiguous_sampling", False, "")
tf.flags.DEFINE_float("learning_rate", 0.5, "learning rate for SGD (default: 0.5)")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
train,dev,test = data_helpers.build_data('fb15k')
x_train,y_train,x_dev,y_dev,x_test,y_test = train[0],train[1],dev[0],dev[1],test[0],test[1]

#Number of entities and relations

n_entities = len(np.unique(np.concatenate((x_train[:,0],x_train[:,2],x_dev[:,0],x_dev[:,2],x_test[:,0], x_test[:,2]))))

n_relations = len(np.unique(np.concatenate((x_train[:,1],x_dev[:,1], x_test[:,1]))))


#logger.info("Nb entities: " + str(self.n_entities))
#logger.info( "Nb relations: " + str(self.n_relations))
#logger.info( "Nb obs triples: " + str(x_train.shape[0]))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        comemb = ComplexLogistic(n_entities,n_relations,n_entities,embedding_size=FLAGS.embedding_dim,l2_reg_lambda=FLAGS.l2_reg_lambda)
        
	# Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdagradOptimizer(0.5)
        grads_and_vars = optimizer.compute_gradients(comemb.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
	
	# Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", comemb.loss)
        acc_summary = tf.scalar_summary("accuracy", comemb.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())


	# Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              comemb.input_x: x_batch,
              comemb.input_y: y_batch,
	      comemb.batch_size_negs:len(x_batch)
            }
            _, step, summaries, loss, mrr,accuracy = sess.run(
                [train_op, global_step, train_summary_op, comemb.loss, comemb.mrr,comemb.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, mrr {:g}, acc {:g}".format(time_str, step, loss, mrr,accuracy))
            train_summary_writer.add_summary(summaries, step)
	    


	def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              comemb.input_x: x_batch,
              comemb.input_y: y_batch,
	      comemb.batch_size_negs : len(x_batch)
            }
            step, summaries, loss, mrr,accuracy = sess.run(
                [global_step, dev_summary_op, comemb.loss, comemb.mrr,comemb.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, mrr {:g}, acc {:g}".format(time_str, step, loss, mrr,accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs,n_entities)
	valid = data_helpers.neg_triples(np.array(list(zip(x_dev,y_dev))),len(x_dev),n_entities)
	print "len of valid",len(valid)
	x_valid,y_valid = zip(*valid)
	print len(x_valid) , len(y_valid)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_valid,y_valid, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        

