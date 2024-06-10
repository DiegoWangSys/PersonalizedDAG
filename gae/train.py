from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf # if v2
#import tensorflow as tf  #if v1
import numpy as np
import scipy.sparse as sp
import pickle

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 800, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 2.')
# hidden2 is the output reduced dimension
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.') # should be gcn_ae here
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data .npy file
#adj, features = load_data(dataset_str) # check here how to load data
usernet_pth = 'UserNet.npy'
userfeat_pth = 'UserFeat.npy'
adj_np = np.load(usernet_pth) # replace the path with data path
                                #here is the User Net data, a symmetric 0/1 n*n matrix numpy narray
                                # n should range 1000~5000, it could not be too large
features = np.load(userfeat_pth)# replace the path with data path 
                                  # here is the User Feature data, a n*30 matrix numpy narray

num_cluster = 2  # here is parameter controlling shape of beta, num_cluster could be 2/3
feat_reduced = np.zeros((features.shape[0],num_cluster))

adj = sp.csr_matrix(adj_np) # here is the sparse adj graph structure



# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# data separation: train / val / test
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# this step will take a long time for sampling

adj = adj_train


if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless features.shape[0] = N (node num)

# Some preprocessing
adj_norm = preprocess_graph(adj)

tf.disable_eager_execution()

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

#features = sparse_to_tuple(features.tocoo())  # here features is a dict

#num_features = features[2][1]  # dim of features
num_features = features.shape[1]
#features_nonzero = features[1].shape[0]
features_nonzero = 0


# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)



# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,embed = model.z_mean)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, epoch,n_cluster,emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
        if epoch == (FLAGS.epochs-1):
            labels = KMeans(n_clusters=n_cluster, random_state=413).fit(emb).labels_
            if num_cluster == 2:
                for i in range(feat_reduced.shape[0]):
                    if labels[i]==0:
                        feat_reduced[i,:] = [-0.5,-0.5]
                    else:
                        feat_reduced[i,:] = [0.5,0.5]
            if num_cluster == 3:
                for i in range(feat_reduced.shape[0]):
                    if labels[i]==0:
                        feat_reduced[i,:] = [1,0]
                    elif labels[i]==1:
                        feat_reduced[i,:] = [-0.5,np.sqrt(3)/2]
                    else:
                        feat_reduced[i,:] = [-0.5,-np.sqrt(3)/2]

            np.save('ReducedFeat.npy',feat_reduced) #save the reduced feature for dag learning
            return None
            #raise Exception("Check here it is embeded features, we load the ReducedFeat in dag_learning.")

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary here is the last features used as dict check placeholders
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false, epoch, n_cluster=num_cluster)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

roc_score, ap_score = get_roc_score(test_edges, test_edges_false, 0,n_cluster=num_cluster)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
