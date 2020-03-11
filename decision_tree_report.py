# Submitted by

# S K Aravind, SXA190006

# Paresh Dashore, PXD190004

# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.



import numpy as np

import os

import graphviz

import math

import matplotlib.pyplot as plt

# tqdm for loading bar

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

# seaborn for plotting heatmap of confusion matrix

import seaborn as sn

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree




def partition(x):
	"""
	Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

	Returns a dictionary of the form
	{ v1: indices of x == v1,
	  v2: indices of x == v2,
	  ...
	  vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
	"""
	Xtrn=x[:,:-1]
	
	k=dict()
	for i in range(len(Xtrn[0])):
		temp=str(i+1)
		unique=set(Xtrn[:,i])
		for j in unique:
			if temp not in k:
				k[temp]=[j]
			else:
				k[temp].append(j)
				
	return k


	# INSERT YOUR CODE HERE

	raise Exception('Function not yet implemented!')





def entropy(y):
	"""
	Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

	Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
	"""
	ytrn = y
	Hz = 0
	for i in set(ytrn):
		try:
			a=(sum(ytrn==i)/len(ytrn))*math.log((sum(ytrn==i)/len(ytrn)),2)
		except Exception as e:
			a=0
		Hz += a
	return Hz



	# INSERT YOUR CODE HERE

	raise Exception('Function not yet implemented!')





def mutual_information(x, y):
	"""
	Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
	over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
	the weighted-average entropy of EACH possible split.

	Returns the mutual information: I(x, y) = H(y) - H(y | x)
	"""

	Hy = x
	Hyx = y

	return Hy-Hyx

	# INSERT YOUR CODE HERE

	raise Exception('Function not yet implemented!')





def add_to_tree(u,k): 
	node=[]
	Hy=-entropy(u[:,-1])
	for i in k:


		for j in k[i]:

			sum1=-1*(entropy(u[u[:,int(i)-1]==j][:,-1]))*(len((u[u[:,int(i)-1]==j]))/len(u))

			sum2=-1*(entropy(u[u[:,int(i)-1]!=j][:,-1]))*(len((u[u[:,int(i)-1]!=j]))/len(u))
			information_gain=mutual_information(Hy,sum1+sum2)
			node.append((i,j,information_gain))


	feature = max(node,key=lambda x:x[2])[0]
	

	feature_node=max(node,key=lambda x:x[2])[1]
	
   
	
	return feature_node,feature


def helper(u,k,max_depth=0):
	## The variable k gives us the attribute value pair
	if max_depth==0:
		return majority(u)
	if purity(u):
		return majority(u)
		
	if len(k)==0:
		return majority(np.column_stack((Xtrn, ytrn)))
	
	
	feature_node,feature=add_to_tree(u,k)
	u_false=u[u[:,int(feature)-1]!=feature_node]
	u_true=u[u[:,int(feature)-1]==feature_node]
	k_false=partition(u_false)
	k_true=partition(u_true)
	
	return {
		(feature,feature_node,False):helper(u_false,k_false,max_depth-1),
		(feature,feature_node,True):helper(u_true,k_true,max_depth-1)
	}
	

def majority(u):
	ytrn=u[:,-1]
	counts = np.bincount(ytrn)
	return np.argmax(counts)

def purity(u):
	ytrn=u[:,-1]
	if len(set(ytrn)) == 1:
		return True
	else:
		return False

	
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
	"""
	Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
	attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
		1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
		2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
		   value of y (majority label)
		3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
	Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
	and partitions the data set based on the values of that attribute before the next recursive call to ID3.

	The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
	to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
	(taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
	attributes with their corresponding values:
	[(x1, a),
	 (x1, b),
	 (x1, c),
	 (x2, d),
	 (x2, e)]
	 If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
	 the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

	The tree is stored as a nested dictionary, where each entry is of the form
					(attribute_index, attribute_value, True/False): subtree
	* The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
	indicates that we test if (x4 == 2) at the current node.
	* The subtree itself can be nested dictionary, or a single label (leaf node).
	* Leaf nodes are (majority) class labels

	Returns a decision tree represented as a nested dictionary, for example
	{(4, 1, False):
		{(0, 1, False):
			{(1, 1, False): 1,
			 (1, 1, True): 0},
		 (0, 1, True):
			{(1, 1, False): 0,
			 (1, 1, True): 1}},
	 (4, 1, True): 1}
	"""
	Xtrn = x
	ytrn = y
	u=np.column_stack((Xtrn, ytrn))

	## The variable k gives us the attribute value pair
	k=partition(u)
	return helper(u,k,max_depth)


	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.

	raise Exception('Function not yet implemented!')





def predict_example(x, tree):
	"""
	Predicts the classification label for a single example x using tree by recursively descending the tree until
	a label/leaf node is reached.

	Returns the predicted label of x according to tree
	"""

	if type(tree) != dict:
		return tree

	feature = int(list(tree.keys())[0][0])-1
	feature_val = list(tree.keys())[0][1]
	if x[feature] == feature_val:
		return predict_example(x, tree[list(tree.keys())[1]])
	else:
		return predict_example(x, tree[list(tree.keys())[0]])



	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.

	raise Exception('Function not yet implemented!')





def compute_error(y_true, y_pred):
	"""
	Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

	Returns the error = (1/n) * sum(y_true != y_pred)
	"""

	# mse
	return (1/len(y_pred)) * sum(y_true != y_pred)

	# INSERT YOUR CODE HERE

	raise Exception('Function not yet implemented!')



def visualize(tree, depth=0):
	"""
	Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
	print the raw nested dictionary representation.
	DO NOT MODIFY THIS FUNCTION!
	"""

	if depth == 0:
		print('TREE')

	for index, split_criterion in enumerate(tree):
		sub_trees = tree[split_criterion]

		# Print the current node: split criterion
		print('|\t' * depth, end='')
		print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

		# Print the children
		if type(sub_trees) is dict:
			visualize(sub_trees, depth + 1)
		else:
			print('|\t' * (depth + 1), end='')
			print('+-- [LABEL = {0}]'.format(sub_trees))





def render_dot_file(dot_string, save_file, image_format='png'):
	"""
	Uses GraphViz to render a dot file. The dot file can be generated using

		* sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn

		* to_graphviz() (function is in this file) for decision trees produced by  your code.

	"""



	# Set path to your GraphViz executable here

	#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

	graph = graphviz.Source(dot_string)

	graph.format = image_format

	graph.render(save_file, view=True)





def to_graphviz(tree, dot_string='', uid=-1, depth=0):
	"""
	Converts a tree to DOT format for use with visualize/GraphViz
	"""

	uid += 1       # Running index of node ids across recursion

	node_id = uid  # Node id of this node

	if depth == 0:

		dot_string += 'digraph TREE {\n'

	for split_criterion in tree:

		sub_trees = tree[split_criterion]

		attribute_index = split_criterion[0]

		attribute_value = split_criterion[1]

		split_decision = split_criterion[2]

		if not split_decision:

			# Alphabetically, False comes first

			dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

		if type(sub_trees) is dict:

			if not split_decision:

				dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)

				dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)

			else:

				dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)

				dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)
		else:

			uid += 1

			dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)

			if not split_decision:

				dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)

			else:

				dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)
	
	if depth == 0:

		dot_string += '}\n'

		return dot_string

	else:

		return dot_string, node_id, uid


if __name__ == '__main__':


	########### part (a) ##############


	for i in range(1,4):
	# Load the training data
		M = np.genfromtxt('./monks-'+str(i)+'.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)

		ytrn = M[:, 0]

		Xtrn = M[:, 1:]

		# Load the test data

		M = np.genfromtxt('./monks-'+str(i)+'.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)

		ytst = M[:, 0]

		Xtst = M[:, 1:]

		train_errors = []
		test_errors = []

		for depth in tqdm(range(1,11)):
			decision_tree = id3(Xtrn, ytrn,  max_depth=depth)
			
			y_pred = [predict_example(x, decision_tree) for x in Xtrn]
			train_errors.append(compute_error(ytrn, y_pred))
			
			y_pred = [predict_example(x, decision_tree) for x in Xtst]
			test_errors.append(compute_error(ytst, y_pred))

		plt.subplot(2,2,i)
		plt.plot([i for i in range(1,11)], train_errors, 'b-', label='Train Error')
		plt.plot([i for i in range(1,11)], test_errors, 'r-', label='Test Error')
		plt.legend()
		plt.xlabel('Depth')
		plt.ylabel('Error')
		plt.title('Monk '+str(i))
	
	plt.tight_layout()
	plt.show()


	# ############## part (b) ################

	M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)

	ytrn = M[:, 0]

	Xtrn = M[:, 1:]

	# Load the test data

	M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)

	ytst = M[:, 0]

	Xtst = M[:, 1:]

	sn.set(font_scale=1.4)
	
	for i in range(1,3):
		decision_tree = id3(Xtrn, ytrn,  max_depth=i)
		y_pred = [predict_example(x, decision_tree) for x in Xtst]
		cf_mat = confusion_matrix(ytst,y_pred)
		dot_str = to_graphviz(decision_tree)
		render_dot_file(dot_str, './my_learned_tree_monks_depth{}'.format(i))
		visualize(decision_tree)
		plt.subplot(1,2,i)
		
		plt.title('Depth : '+str(i))
		sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

	plt.suptitle('Confusion Matrix (Our Tree, Monks)', fontsize=12)
	plt.tight_layout()
	plt.show()

	


	############  part (c)  ################

	M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)

	ytrn = M[:, 0]

	Xtrn = M[:, 1:]

	# Load the test data

	M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)

	ytst = M[:, 0]

	Xtst = M[:, 1:]

	dtc = DecisionTreeClassifier(max_depth=3)
	dtc.fit(Xtrn, ytrn)
	y_pred = dtc.predict(Xtst)

	tst_err = compute_error(ytst, y_pred)

	print('Test Error from SKLearn = {0:4.2f}%.'.format(tst_err * 100))
	cf_mat = confusion_matrix(ytst,y_pred)
	
		
	plt.title('Depth : '+str(3))
	sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

	plt.title('Confusion Matrix (Sklearn, Monks)', fontsize=12)
	plt.tight_layout()
	plt.show()

	render_dot_file(tree.export_graphviz(dtc), './sklearn_tree_monks')


	############ part (d) ################

	M = np.genfromtxt('./tic-tac-toe.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)


	ytrn = M[:, 0]

	Xtrn = M[:, 1:]

	M = np.genfromtxt('./tic-tac-toe.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)


	ytst = M[:, 0]

	Xtst = M[:, 1:]

	sn.set(font_scale=1.4)
	
	for i in range(1,3):
		decision_tree = id3(Xtrn, ytrn,  max_depth=i)
		y_pred = [predict_example(x, decision_tree) for x in Xtst]
		cf_mat = confusion_matrix(ytst,y_pred)
		dot_str = to_graphviz(decision_tree)
		render_dot_file(dot_str, './my_learned_tree_tic_tac_toe_depth{}'.format(i))
		plt.subplot(1,2,i)
		
		plt.title('Depth : '+str(i))
		sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

	plt.suptitle('Confusion Matrix (Our Tree, Tic Tac Toe)', fontsize=12)
	plt.tight_layout()
	plt.show()


	dtc = DecisionTreeClassifier(max_depth=3)
	dtc.fit(Xtrn, ytrn)
	y_pred = dtc.predict(Xtst)

	tst_err = compute_error(ytst, y_pred)
	
	cf_mat = confusion_matrix(ytst,y_pred)
	
		
	plt.title('Depth : '+str(3))
	sn.heatmap(cf_mat, annot=True, annot_kws={"size": 16}, fmt='g')

	plt.title('Confusion Matrix (Sklearn, Tic Tac Toe)', fontsize=12)
	plt.tight_layout()
	plt.show()

	render_dot_file(tree.export_graphviz(dtc), './sklearn_tree_tic_tac_toe')




# 	# Pretty print it to console

# 	visualize(decision_tree)


# # #     # Visualize the tree and save it as a PNG image

	# dot_str = to_graphviz(decision_tree)

	# render_dot_file(dot_str, './my_learned_tree')



# # #     # Compute the test error

# 	y_pred = [predict_example(x, decision_tree) for x in Xtst]

# 	tst_err = compute_error(ytst, y_pred)

# 	print('Test Error = {0:4.2f}%.'.format(tst_err * 100))



