from __future__ import division, print_function, absolute_import
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import tflearn
import numpy as np
import tensorflow as tf
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT, NUM_DICTIONARY_WORDS,\
    SentenceToVector
np.set_printoptions(threshold=np.inf)

TRAIN = 1

if TRAIN:
    data_path_pos = "D:\\datasets\\NLPLab\\data\\aclImdb_v1\\aclImdb\\train\\pos\\pos_vec\\"
    data_path_neg = "D:\\datasets\\NLPLab\\data\\aclImdb_v1\\aclImdb\\train\\neg\\neg_vec\\"
    
    #10000 samples
    sample_count = 10000
    s = 100
    featuresx = np.array([])
    labels = np.array([])
    while s <= sample_count:
        features_pos = np.loadtxt(data_path_pos + 'pos_vectors{}.txt'.format(s), dtype = int)
        #features_pos = features_pos.reshape((100, 10, 10, 3))
        #print(features_pos.shape)
        features_neg = np.loadtxt(data_path_neg + 'neg_vectors{}.txt'.format(s), dtype = int)
        #features_neg = features_neg.reshape((100, 10, 10, 3))
        #print(features_neg.shape)
        featuresx = np.append(featuresx, features_pos)
        featuresx = np.append(featuresx, features_neg)
        labels = np.append(labels, np.ones((100,)))
        labels = np.append(labels, np.zeros((100,)))
        s += 100
        #print(featuresx.shape)
        #print(s)
    
    features = featuresx.reshape((sample_count * 2, 10, 10, 3))
    print(features.shape)
    print(labels.shape)
    
    train_split_proportion = 0.9
    split = int(len(features) * train_split_proportion)
    trainX, testX = features[:split], features[split:]
    trainY, testY = labels[:split], labels[split:]
    trainY = to_categorical(trainY, 2)
    testY = to_categorical(testY, 2)

tf.reset_default_graph()
tflearn.config.init_training_mode()
tflearn.init_graph(soft_placement=True)

net = input_data(shape=[None, 10, 10, 3])
net = conv_2d(net, 10, 3, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 20, 3, activation='relu')
net = conv_2d(net, 20, 3, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 128, activation='relu')
net = dropout(net, 0.3)
net = fully_connected(net, 2, activation='softmax')
net = regression(net, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)

CNN_model = tflearn.DNN(net, tensorboard_verbose=0)

if TRAIN:
    with tf.device('/device:GPU:0'):
        CNN_model.fit(trainX, trainY, n_epoch=250, shuffle=True, validation_set=(testX, testY),
                  show_metric=True, batch_size=100)
        CNN_model.save("model_CNN_2classes.tfl")
else:
    print('loading CNN...')
    CNN_model.load("model_CNN_2classes.tfl")

