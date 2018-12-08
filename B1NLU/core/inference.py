# implementing the inference logic to classify the detected intents and extract the slots/entities
# we train a deep neural network (DNN) on a data set of intents. save it to b1_intents_model_v1. the training code is not included here,
# instead we load and make use of the DNN model to apply intent classification on the input sentence


# import required libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from api.core import b1NER as bner
import numpy as np
import pickle
import tflearn
import tensorflow as tf
import random
import os
import warnings
import json




# Just disables the warning, doesn't enable AVX/FMA
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.resetwarnings()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

# load the training data
pwd = os.getcwd()
pwd = os.getcwd()
data = pickle.load(open(pwd+"/data/training_data","rb"))

# bag of words
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import intents file
pwd = os.getcwd()
# load the intents file
with open(pwd+'/data/B1_intents_dataset.json') as json_data:
  intents = json.load(json_data)


# Build a deep neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
# Define a model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# load our saved model
model.load(pwd+'/models/b1_intents_model/model.tflearn')

# cleanup
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.25

# intent classification procedure
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r >ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list



# given some intent, it returns the probability/accuray of intent prediction
ERROR_THRESHOLD = 0.25
def accuracy(intent, sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    acc = 0
    for r in results:
        if classes[r[0]] == intent:
            return_list.append((r[1]))
            acc = r[1]
            print ("acc: ", acc)

    return acc

# extract the robot response to an intent
def getIntentResponse(intent, sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == intent:
                    # a random response from the intent
                    if intent == "Give discharge instructions":
                        return i['responses']
                    else:
                        return (random.choice(i['responses']))

            results.pop(0)

# get top intent
def getTopIntent(results):
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return i['tag']

# build a unified structure of data object contains the the detected intent and slots
def infer(text):
    #text to intents
    intents = classify(text)
    # take the top intent
    intent = getTopIntent(intents)
    # a temporary list to save the items
    bdata = {}
    # a container list to return the b1 data object
    b1data = {}
    b1data['intent'] = intent
    # extract slots from text
    slots = bner.getB1Entities(text)
    #add slots to the list
    b1data.update(slots)
    # add B1 response to the list



    if 'FOOD.CATEGORY' in b1data or 'RESTAURANT.NAME' in b1data:
        b1data['intent'] = "OrderFood"
    rest = ""
    food = ""
    for k, v in b1data.items():
             if k == "RESTAURANT.NAME":
                rest = v
             if k == "FOOD.CATEGORY":
                food = v[0]

    if "FOOD.CATEGORY" in b1data and "RESTAURANT.NAME" in b1data:
        b1data['B1_response'] = "OK, here are options of "+ food +" from "+ rest

    elif "FOOD.CATEGORY" in b1data and "RESTAURANT.NAME" not in b1data:
        b1data['B1_response'] = "OK, here are options of " + food

    elif "RESTAURANT.NAME" in b1data and "FOOD.CATEGORY" not in b1data:
        b1data['B1_response'] = "OK, here are options from " + rest

    elif "RESTAURANT.NAME" not in b1data and "FOOD.CATEGORY" not in b1data:
        b1data['B1_response'] = "What are you in the mood for?"

    else:
        b1data['B1_response'] = getIntentResponse(intent, text)




    # return the list
    return b1data







