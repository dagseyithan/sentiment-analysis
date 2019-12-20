'''
Created on 22 Feb 2018

@author: seyit
'''
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT,\
    SentenceToVector, RemovePunctuationAndCorrectSpelling,\
    SentenceToVectorWithoutPadding
from labprojecttf.preprocessorLSTM import stripped_sentences
    
PROCESS = 2

data_path = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\"

if PROCESS == 0:
    vectors = np.loadtxt(data_path + 'combined_data\\sentence_vectors.txt', dtype = int)
    labels = np.loadtxt(data_path + 'combined_data\\sentence_labels.txt', dtype = int)
    negatives = []
    positives = []
    for vector, label in zip(vectors, labels):
        if label == 0:
            negatives.append(vector)
        else:
            positives.append(vector)
        
if PROCESS == 1:
    with open(data_path + "combined_data\\opinion_stripped_sentences.txt", 'r') as f:
        stripped_sentences = f.readlines()
    stripped_sentences_asint = []
    for sentence in stripped_sentences:
        stripped_sentences_asint.append(SentenceToVectorWithoutPadding(sentence))
    labels = np.loadtxt(data_path + 'combined_data\\sentence_labels.txt', dtype = int)
    negatives = []
    positives = []
    for vector, label in zip(stripped_sentences_asint, labels):
        if vector != []:
            if label == 0:
                negatives.append(vector)
            else:
                positives.append(vector)
    
    negatives = np.array(negatives)
    negatives = [np.trim_zeros(vector) for vector in negatives]
    flat_negatives = [item for sublist in negatives for item in sublist]
    lengths_negatives = [len(vector) for vector in negatives]
    lengths_negatives = np.array(lengths_negatives)
    negatives = np.array(flat_negatives)
    negatives = negatives.flatten().reshape(-1,1)
    
    positives = np.array(positives)
    positives = [np.trim_zeros(vector) for vector in positives]
    flat_positives = [item for sublist in positives for item in sublist]
    lengths_positives = [len(vector) for vector in positives]
    lengths_positives = np.array(lengths_positives)
    positives = np.array(flat_positives)
    positives = positives.flatten().reshape(-1,1)

    HMM_positive = hmm.GaussianHMM(20, covariance_type="full", n_iter=12, verbose=True)
    HMM_positive.fit(positives, lengths_positives)
    joblib.dump(HMM_positive, data_path + 'combined_data\\HMM_positive.pkl')
    HMM_negative = hmm.GaussianHMM(20, covariance_type="full", n_iter=12, verbose=True)
    HMM_negative.fit(negatives, lengths_negatives)
    joblib.dump(HMM_negative, data_path + 'combined_data\\HMM_negative.pkl')
    
'''        
lengths_negatives = np.ones((len(negatives),), dtype = int)
lengths_positives = np.ones((len(positives),), dtype = int)
lengths_negatives = MAX_SENTENCE_LENGHT * lengths_negatives
lengths_positives = MAX_SENTENCE_LENGHT * lengths_positives
'''


HMM_negative = joblib.load(data_path + 'combined_data\\HMM_negative.pkl')
HMM_positive = joblib.load(data_path + 'combined_data\\HMM_positive.pkl')

test_sentence = "excellent"


def GetHMMVote(sentence):
    sentence = SentenceToVectorWithoutPadding(sentence)
    sentence = sentence.reshape(-1,1)
    pos = HMM_positive.score(sentence)
    neg = HMM_negative.score(sentence)
    print(pos, neg)
    if pos > neg:
        return 1
    else:
        return 0 

print(GetHMMVote(test_sentence)) 
