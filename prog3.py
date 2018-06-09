#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import sys


# prettyPrint: prettyPrint to stderr
def prettyPrint(iter, train, dev):
    sys.stderr.write('Iter %06d: train=%.3f dev=%.3f\n' % (iter, train, dev))
    return None


# isNumber: does what it says on the tin
def isNumber(i):
    try:
        float(i)
    except ValueError:
        return False
    return True


def toOneHot(X, dim):
    '''
    :param X: Vector of indices
    :param dim: Dimension of indexing
    :return: Matrix of one hots
    '''
    # empty one-hot matrix
    hotmatrix = np.zeros((X.shape[0], dim))
    # fill indice positions
    hotmatrix[np.arange(X.shape[0]), X.astype(int)] = 1
    return hotmatrix


'''
#   multilayer: Method to handle arbitrarily deep neural networks (from 1 to nLayers)
#
#       Preconditions:
#       - x is an already initialized placeholder of proper dimmensions
#       - hidden_act == sig, tanh or relu (for optimizer)
#
#       Postconditions:
#       - Data has been passed through at least 1 hidden layer
#       - Returns the output of the final hidden layer + bias
'''
def multiLayer(x, dim, num_classes, hidden_units, init_range,  hidden_act, nlayers):

    h1_weight = tf.Variable(tf.random_uniform(shape=[dim, hidden_units], minval=-init_range, maxval=init_range, dtype=tf.float32))
    hn_weight = tf.Variable(tf.random_uniform(shape=[hidden_units, hidden_units], minval=-init_range, maxval=init_range, dtype=tf.float32))
    out_weight = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes], minval=-init_range, maxval=init_range, dtype=tf.float32))

    h_bias = tf.Variable(tf.random_uniform(shape=[hidden_units], minval=-init_range, maxval=init_range, dtype=tf.float32))
    out_bias = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-init_range, maxval=init_range, dtype=tf.float32))

    if (hidden_act == "sig"):
        layer_n_old = tf.nn.sigmoid(tf.add(tf.matmul(x, h1_weight), h_bias)) #Hidden layer with SIG activation
        for i in range(0, nlayers-1):
            layer_n = tf.nn.sigmoid(tf.add(tf.matmul(layer_n_old, hn_weight), h_bias))
            layer_n_old = layer_n

    elif (hidden_act == "tanh"):
        layer_n_old = tf.nn.tanh(tf.add(tf.matmul(x, h1_weight), h_bias)) #Hidden layer with TANH activation
        for i in range(0, nlayers-1):
            layer_n = tf.nn.tanh(tf.add(tf.matmul(layer_n_old, hn_weight), h_bias))
            layer_n_old = layer_n

    elif (hidden_act == "relu"):
        layer_n_old = tf.nn.relu(tf.add(tf.matmul(x, h1_weight), h_bias)) #Hidden layer with RELU activation
        for i in range(0, nlayers-1):
            layer_n = tf.nn.relu(tf.add(tf.matmul(layer_n_old, hn_weight), h_bias))
            layer_n_old = layer_n
    return tf.matmul(layer_n_old, out_weight) + out_bias


'''
#   classifyOrRegress: uses an arbitrarily deep neural network to evaluate data
#
#       Preconditions:
#       - args has been verified and populated
#       - Target feature files have N lines of D datapoints
#       - If regression mode is set:
#           - Target files have N lines (Ntarget = Nfeat) with C datapoints
#           - Num_classes will be ignored if specified
#           - Evaluation will use MSE
#       - If classification mode is set:
#           - Target files have N lines of 1 datapoint
#           - Num_classes must be set, program will not run otherwise
#           - Evaluation uses Softmax
#
#
#       Postconditions:
#       - Iteration number and accuracy ratings are printed to stderr
#       - Neural network training / evaluation will run for the given number of epochs
'''
def classifyOrRegress(args):
    train_feat = np.loadtxt(open(args["-train_feat"], 'r'), ndmin=2, dtype=float)
    dev_feat = np.loadtxt(open(args["-dev_feat"], 'r'), ndmin=2, dtype=float)

    nTrain, dim = np.shape(train_feat)

    # Opens the feature files and sets num_classes
    if "-num_classes" in args:  # Classification setup
        num_classes = int(args["-num_classes"])
        train_target = toOneHot(np.loadtxt(open(args["-train_target"], 'r')), num_classes)
        dev_target = toOneHot(np.loadtxt(open(args["-dev_target"], 'r')), num_classes)
    else:   # Regression setup
        train_target = np.loadtxt(open(args["-train_target"], 'r'), ndmin=2)
        dev_target = np.loadtxt(open(args["-dev_target"], 'r'), ndmin=2)
        nTrain, num_classes = np.shape(train_target)

    hidden_units = int(args["-nunits"])
    init_range = float(args["-init_range"])
    learn_rate = float(args["-learnrate"])
    # Sets batch_size to specified minibatch size, or the total number of training points without minibatching
    if "-mb" in args and int(args["-mb"]) > 0:
        batch_size = int(args["-mb"])
    else:
        batch_size = nTrain

    x = tf.placeholder(tf.float32, [None, dim])

    # Hidden layer setup
    if "-nlayers" in args and int(args["-nlayers"]) > 1: # Uses user-supplied number of layers
        pred = multiLayer(x, dim, num_classes, hidden_units, init_range, args["-hidden_act"], int(args["-nlayers"]))
    else:   # Default of 1 layer
        pred = multiLayer(x, dim, num_classes, hidden_units, init_range, args["-hidden_act"], 1)

    # Placeholder for correct outputs
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    if args["-type"].lower() == 'c':
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
    else:
        cost = tf.reduce_mean(tf.pow(pred-y_, 2))

    if args["-optimizer"] == "adam":
        optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(int(args["-epochs"])):
        if batch_size == nTrain: # regular training on whole dataset
            batch_xs = train_feat
            batch_ys = train_target
            sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
        else: # Minibatching, if dataset is not evenly divisible by batch_size, skips last partial batch
            batch_index = 0
            while (batch_index + batch_size) < nTrain:
                batch_xs = train_feat[batch_index:(batch_index+batch_size)]
                batch_ys = train_target[batch_index:(batch_index+batch_size)]
                sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
                batch_index += batch_size + 1

        if args["-type"].lower() == 'c':
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            accuracy = cost

        # Evaluates the model's accuracy on the two datasets
        train_acc = sess.run(accuracy, feed_dict={x: train_feat, y_: train_target})
        dev_acc = sess.run(accuracy, feed_dict={x: dev_feat, y_: dev_target})
        prettyPrint(i+1, train_acc, dev_acc)

    return True


'''
#   verifyArgs: verifies that the essential arguments have been read in, and the values are in the correct range
#
#       Preconditions:
#       - dict is initialized
#
#       Postconditions:
#       - Returns true if all essential arguments are passed in, and hold valid values, false otherwise
#       - Prints message to stderr for the first missing or out of range essential argument
'''
def verifyArgs(dict):
    if "-train_feat" not in dict:
        sys.stderr.write("Error: missing -train_feat argument")
        return False
    if "-train_target" not in dict:
        sys.stderr.write("Error: missing -train_target argument")
        return False
    if "-dev_feat" not in dict:
        sys.stderr.write("Error: missing -dev_feat argument")
        return False
    if "-dev_target" not in dict:
        sys.stderr.write("Error: missing -dev_target argument")
        return False
    if "-epochs" not in dict:
        sys.stderr.write("Error: missing -epochs argument")
        return False
    if "-learnrate" not in dict or not isNumber(dict["-learnrate"]):
        sys.stderr.write("Error: missing -learnrate argument")
        return False
    if "-nunits" not in dict or not isNumber(dict["-nunits"]):
        sys.stderr.write("Error: missing -nunits argument")
        return False
    if "-hidden_act" not in dict or not (dict['-hidden_act'].lower() == 'sig' or dict['-hidden_act'].lower() == 'tanh' or dict['-hidden_act'].lower() == 'relu'):
        sys.stderr.write("Error: missing or invalid -hidden_act argument")
        return False
    if "-optimizer" not in dict or not (dict['-optimizer'].lower() == 'adam' or dict['-optimizer'].lower() == 'grad'):
        sys.stderr.write("Error: missing or invalid -optimizer argument")
        return False
    if "-init_range" not in dict:
        sys.stderr.write("Error: missing -init_range argument")
        return False
    if "-type" not in dict:
        sys.stderr.write("Error: missing -type argument")
        return False
    if dict['-type'].lower() != 'r':
        if (dict['-type'].lower() == 'c' and "-num_classes" not in dict):
            sys.stderr.write("Error: invalid -type argument or missing -num_classes argument")
            return False
    return True


# Handles input arguments, then passes off to the tensorflow main body
def main():
    length = len(sys.argv)
    argsDict = {}
    if length % 2 != 0:
        for i in range(1, length, 2):
            argsDict[sys.argv[i]] = sys.argv[i+1]
    else:
        sys.stderr.write("Error: invalid number of args")
    if verifyArgs(argsDict):
        classifyOrRegress(argsDict)
    else:
        print("Missing one or more essential program arguments, shutting down")
    return None

main()
