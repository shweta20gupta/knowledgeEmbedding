#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import datetime
import data_helpers
from complex_embeddings import ComplexLogistic

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data of missing entities

missing_entities = open("./missing_entities.csv",'r').readlines()


# load relation and entities dictionaries
with open("./entities_indexes.txt","rb") as f:
	entities_indexes = pickle.load(f)

with open("./relations_indexes.txt","rb") as f:
	relation_indexes = pickle.load(f)
relation_names = {v:k for k,v in relation_indexes.iteritems()}
entities_names = {v:k for k,v in entities_indexes.iteritems()}


checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
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
	batch_size_negs = graph.get_operation_by_name("batch_size_negs").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/probabilities").outputs[0]





# generate test data for each missing entry"
def link_prediction(sub,rel,obj,entities_indexes,relation_indexes):
	x_test = np.empty((0,3), int)
	if sub == None:
		for ent in entities_indexes: 
    			x_test_sub = np.array([entities_indexes[ent],relation_indexes[rel],entities_indexes[obj]])
    			x_test = np.vstack((x_test,x_test_sub))
	elif obj == None:
		for ent in entities_indexes:
			x_test_obj = np.array([entities_indexes[sub],relation_indexes[rel],entities_indexes[ent]])
			x_test = np.vstack((x_test,x_test_obj))
	return x_test

for line in missing_entities:
	#line = line.strip().split("\t")
	#sub,rel,obj = line[0],line[1],line[2]
	sub = None
	rel = '/location/country/form_of_government'
	obj = '/m/06cx9'
	x_test = link_prediction(sub,rel,obj,entities_indexes,relation_indexes)
	link_predictions = sess.run(predictions, {input_x: x_test, batch_size_negs:len(x_test)})
	link_predictions = np.array(link_predictions)
	ind = link_predictions.argsort()[-10:][::-1]
	predicted_entities = x_test[ind]
	if obj == None:
		obj_pred_ind = [i[2] for i in predicted_entities]
		obj_pred = [entities_names[i] for i in obj_pred_ind]
		print "predicted objects for ", sub,rel, "are" , obj_pred
	if sub == None:
                sub_pred_ind = [i[0] for i in predicted_entities]
                sub_pred = [entities_names[i] for i in sub_pred_ind]
                print "predicted subjects for ", rel,obj ,"are" , sub_pred

		





















