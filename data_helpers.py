import scipy
import scipy.io
import random
import pickle
import numpy as np


def parse_line(line):
	line = line.strip().split("\t")
	sub,rel,obj,val  = line[0],line[1],line[2],1

	return sub,obj,rel,val


def load_triples_from_txt(filenames, entities_indexes = None, relations_indexes = None):
	"""
	Take a list of file names and build the corresponding dictionary of triples
	"""


	if entities_indexes == None:
		entities_indexes= dict()
		entities = set()
		next_ent = 0
	else:
		entities = set(entities_indexes)
		next_ent = max(entities_indexes.values()) + 1


	if relations_indexes == None:
		relations_indexes= dict()
		relations= set()
		next_rel = 0
	else:
		relations = set(relations_indexes)
		next_rel = max(relations_indexes.values()) + 1

	data = dict()



	for filename in filenames:
		with open(filename) as f:
			lines = f.readlines()

		for line in lines:

			sub,obj,rel,val = parse_line(line)


			if sub in entities:
				sub_ind = entities_indexes[sub]
			else:
				sub_ind = next_ent
				next_ent += 1
				entities_indexes[sub] = sub_ind
				entities.add(sub)
				
			if obj in entities:
				obj_ind = entities_indexes[obj]
			else:
				obj_ind = next_ent
				next_ent += 1
				entities_indexes[obj] = obj_ind
				entities.add(obj)
				
			if rel in relations:
				rel_ind = relations_indexes[rel]
			else:
				rel_ind = next_rel
				next_rel += 1
				relations_indexes[rel] = rel_ind
				relations.add(rel)

			data[ (sub_ind, rel_ind, obj_ind)] = val

	return data, entities_indexes, relations_indexes


def build_data(name, path = './datasets'):


	folder = path + '/' + name + '/'


	train_triples, entities_indexes, relations_indexes = load_triples_from_txt([folder + 'train.txt'])


	valid_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'valid.txt'],
					entities_indexes = entities_indexes , relations_indexes = relations_indexes)

	test_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'test.txt'],
					entities_indexes = entities_indexes, relations_indexes = relations_indexes)


	train = np.array(list(train_triples.keys())).astype(np.int64), np.array(list(train_triples.values())).astype(np.float32)
	valid = np.array(list(valid_triples.keys())).astype(np.int64), np.array(list(valid_triples.values())).astype(np.float32)
	test = np.array(list(test_triples.keys())).astype(np.int64), np.array(list(test_triples.values())).astype(np.float32)

	with open("./entities_indexes.txt","wb") as f:
		pickle.dump(entities_indexes,f)
	
	with open("./relations_indexes.txt","wb") as f:
                pickle.dump(relations_indexes,f)
	
	return train,valid,test



def batch_iter(data, batch_size, num_epochs, n_entities,shuffle=True,neg_ratio=2,test=False):
    
   	# Generates a batch iterator for a dataset.
    
	data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
		for batch_num in range(num_batches_per_epoch):
            		start_index = batch_num * batch_size
            		end_index = min((batch_num + 1) * batch_size, data_size)
            		batch_data = data[start_index:end_index]
			batch_len = len(batch_data)
			if not test:
				shuffled_data = neg_triples(batch_data,batch_len,n_entities,shuffle=True,neg_ratio=2)
				
			else:
				shuffled_data = test_triples(data,batch_data,batch_len,n_entities)
			yield shuffled_data

				
def test_triples(data,batch_data,batch_size,n_entities):
	sub_data = batch_data
	obj_data = batch_data
	data = convert_to_dict(data)
	for i in range(batch_size):
		for ind in range(n_entities):
			sub_data[i][0][0]= ind
			obj_data[i][0][2] = ind
			sub_data[i][1] = 1 if tuple(sub_data[i][0]) in data.keys() else 0
			obj_data[i][1] = 1 if tuple(obj_data[i][0]) in data.keys() else 0
			batch_data = np.vstack((batch_data,sub_data[i],obj_data[i]))
	return batch_data

def convert_to_dict(data):
	data_dict = dict()
	x,y = zip(*data)
	x = map(tuple,x)
	for i,key in enumerate(x):
		data_dict[key]=y[i]
	return data_dict

	   		
def neg_triples(batch_data,batch_size,n_entities,shuffle=True,neg_ratio=2):
	neg_triples = batch_data
	rdm_entities = np.random.randint(0,n_entities,batch_size*neg_ratio)
        rdm_choices = np.random.random(batch_size*neg_ratio)

	for i in range(batch_size):
		for j in range(neg_ratio):
			cur_ind = i* neg_ratio + j
			if rdm_choices[cur_ind] < 0.5:
				neg_triples[i][0][0] = rdm_entities[cur_ind]
			else:
				neg_triples[i][0][2] = rdm_entities[cur_ind]
			neg_triples[i][1] = 0
			batch_data = np.vstack((batch_data,neg_triples[i]))
	if shuffle:
        	shuffle_indices = np.random.permutation(np.arange(len(batch_data)))
            	shuffled_data = batch_data[shuffle_indices]
        else:
            	shuffled_data = batch_data

	return shuffled_data

	
'''
#Testing ...
train,valid,test = build_data('fb15k')
#print len(x[0]),len(y[0]),len(z[0])
train_indexes = train[0]
valid_indexes = valid[0]
test_indexes = test[0]
n_entities = len(np.unique(np.concatenate((train_indexes[:,0], train_indexes[:,2], valid_indexes[:,0], valid_indexes[:,2], test_indexes[:,0], test_indexes[:,2]))))
batches = batch_iter(list(zip(train[0],train[1])),20,1,n_entities)
for batch in batches:
	print len(batch)

'''
