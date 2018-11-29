import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from sklearn.externals import joblib
import time 
import sys
# we're still going to use a Tokenizer here, but we don't need to fit it
t1 = time.time()
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)
tokenizer = Tokenizer(num_words=4090)
# for human-friendly printing
labels =  ['Desktop','Ebook Reader','Mobile Phone','TV Device','Tablet',
 'Windows Desktop','general Desktop','general Mobile Phone']
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
    return wordIndices
# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')
label_encoder = joblib.load('labelEncoder.h5')
import time
t1 = time.time()
userAgent = sys.argv[1]
device  = sys.argv[2]
trainY = label_encoder.transform([device])
trainY = keras.utils.to_categorical(trainY,8)

# okay here's the interactive part
model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# format your input for the neural net
testArr = convert_text_to_index_array(userAgent)
input1 = tokenizer.sequences_to_matrix([testArr], mode='binary')
print (input1.shape)
# predict which bucket your input belongs in
history = model.fit([input1], [trainY],
    batch_size=32,
    epochs=5,
    verbose=1,
    shuffle=False)
t2 = time.time()
print (t2-t1)
model.save_weights('model.h5')
t2 = time.time()
print ("time:",t2-t1)
# and print it for the humons
