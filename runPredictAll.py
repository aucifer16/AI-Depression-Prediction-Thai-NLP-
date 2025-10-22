import nltk
from nltk import NaiveBayesClassifier as nbc
from nltk.classify import accuracy
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
import pickle
#from pythainlp.transliterate import romanize
#from pythainlp.translate import download_model_all
from pythainlp.summarize import summarize
import pandas as pd

print("Start Program")


f = open('finalized_model.pickle', 'rb')
classifier = pickle.load(f)
f.close()

f = open('vocabulary_model.pickle', 'rb')
vocabulary = pickle.load(f)
f.close()
print("Run Predict")
x=0
def mypredit(fname):
    global x
    x =x+1
    print(x)
    test_sentence = fname
    featurized_test_sentence={i:(i in word_tokenize(test_sentence.lower()))for i in vocabulary}
    return classifier.classify(featurized_test_sentence)
    #return fname+"vvv"
   #print('tag :',classifier.classify(featurized_test_sentence))
df = pd.read_csv('usertwitter_iamsobad15feed.csv')
df.head()
df['Result'] = df['text'].apply(mypredit)
df.to_csv('usertwitter_iamsobad15feed_Predit.csv')
print("ok")


