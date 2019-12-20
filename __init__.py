#from labprojecttf.trainCNN import CNN_model
#from labprojecttf.constructsentimentmaps import ConstructSentimentMaps
#import numpy as np

#def GetReviewPolarity(review):
    #sentiment_map = ConstructSentimentMaps(review)
    #smap = np.dstack([sentiment_map['pos'], sentiment_map['neg'], sentiment_map['so']]).reshape([1, 10, 10, 3])
    #prediction = CNN_model.predict_label(smap)
    #return prediction[0,0]
