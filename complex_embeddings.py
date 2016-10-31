"""
 Define Complex Logisitic Model for learning Embeddings.
"""
import tensorflow as tf
import numpy as np




class ComplexLogistic(object):

	def __init__(self,no_subjects,no_objects,no_relations,embedding_size,l2_reg_lambda):
		
		# Placeholders for input, output and dropout
        	self.input_x = tf.placeholder(tf.int32,name="input_x")
	        self.input_y = tf.placeholder(tf.float32,name="input_y")
		self.batch_size_negs = tf.placeholder(tf.int32,name="batch_size_negs")
		

	        # Keeping track of l2 regularization loss (optional)
        	l2_loss = tf.constant(0.0)

       		 # Embedding layer
        	with tf.device('/cpu:0'), tf.name_scope("embedding"):
            		e1 = tf.Variable(tf.random_uniform([max(no_subjects,no_objects),embedding_size], -1.0, 1.0),name="e1")
			e2 = tf.Variable(tf.random_uniform([max(no_subjects,no_objects), embedding_size], -1.0, 1.0),name="e2")
                	r1 = tf.Variable(tf.random_uniform([no_relations, embedding_size], -1.0, 1.0),name="r1")
                	r2 = tf.Variable(tf.random_uniform([no_relations, embedding_size], -1.0, 1.0),name="r2")
			self.e1_sub = tf.reshape(tf.nn.embedding_lookup(e1,tf.gather(tf.transpose(self.input_x),[0])),tf.pack([self.batch_size_negs,embedding_size]))
			self.e1_obj = tf.reshape(tf.nn.embedding_lookup(e1,tf.gather(tf.transpose(self.input_x),[2])),tf.pack([self.batch_size_negs,embedding_size]))  
			self.e2_sub = tf.reshape(tf.nn.embedding_lookup(e2,tf.gather(tf.transpose(self.input_x),[0])),tf.pack([self.batch_size_negs,embedding_size]))
	                self.e2_obj = tf.reshape(tf.nn.embedding_lookup(e2,tf.gather(tf.transpose(self.input_x),[2])),tf.pack([self.batch_size_negs,embedding_size]))
			self.r1 = tf.reshape(tf.nn.embedding_lookup(r1,tf.gather(tf.transpose(self.input_x),[1])),tf.pack([self.batch_size_negs,embedding_size]))
                        self.r2 = tf.reshape(tf.nn.embedding_lookup(r2,tf.gather(tf.transpose(self.input_x),[1])),tf.pack([self.batch_size_negs,embedding_size]))  
			
		with tf.name_scope("output"):
			self.scores = tf.reduce_sum((self.e1_sub*self.r1*self.e1_obj),1)\
			+ tf.reduce_sum((self.e2_sub*self.r1*self.e2_obj),1)\
			+ tf.reduce_sum((self.e1_sub*self.r2*self.e2_obj),1)\
			- tf.reduce_sum((self.e2_sub*self.r2*self.e1_obj),1)
			self.probabilities = tf.sigmoid(self.scores,name = "probabilities")
			self.predictions = tf.round(self.probabilities,name = "predictions")           	

       		 # Calculate Mean  loss
       		with tf.name_scope("loss"):
	    		l2_loss += tf.nn.l2_loss(self.e1_sub)
	   		l2_loss += tf.nn.l2_loss(self.e1_obj)
	    		l2_loss += tf.nn.l2_loss(self.e2_sub)
	    		l2_loss += tf.nn.l2_loss(self.e2_obj)
	    		l2_loss += tf.nn.l2_loss(self.r1)
	    		l2_loss += tf.nn.l2_loss(self.r2)	
            		losses = tf.nn.softplus(-self.input_y*self.predictions)
            		self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
	
        	# Accuracy
		
        	with tf.name_scope("mrr"):
			correct_predictions = tf.cast(tf.equal(self.predictions,self.input_y),"float")
                        indices_false = tf.not_equal(self.predictions,self.input_y)
			                       

                        idx_false = tf.reshape(tf.where(indices_false),[-1])
			idx_true = tf.reshape(tf.where(tf.logical_not(indices_false)),[-1])

			pred_false = tf.gather(correct_predictions, idx_false)
			pred_true = tf.gather(correct_predictions,idx_true)
			pred_false = 0.5 + pred_false
			rank = tf.concat(0,[pred_true,pred_false])
                        self.mrr = tf.reduce_mean(tf.cast(rank, "float"), name="mrr")


		with tf.name_scope("accuracy"):
			self.accuracy = tf.reduce_mean(correct_predictions,name="accuracy")



                                                          
