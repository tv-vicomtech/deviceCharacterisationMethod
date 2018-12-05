import json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import csv

# we're still going to use a Tokenizer here, but we don't need to fit it
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)
tokenizer = Tokenizer(num_words=3500)
# for human-friendly printing
labels = ['Desktop', 'Mobile Phone', 'TV Device', 'Tablet']


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

# okay here's the interactive part

# format your input for the neural net
fo = open("userAgents5.csv", "r")
lines = fo.readlines()

with open('results_neural5.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    correct = 0
    cnt = 0
    for x in range(1, len(lines)):
        cnt += 1
        if x <= 1000:
            real_val = 'Mobile Phone'
        elif x <= 2000:
            real_val = 'Tablet'
        elif x <= 3000:
            real_val = 'Desktop'
        else:
            real_val = 'TV Device'
        userAgent = lines[x]
        testArr = convert_text_to_index_array(userAgent)
        inp = tokenizer.sequences_to_matrix([testArr], mode='binary')
        # print(testArr)
        # predict which bucket your input belongs in
        pred = model.predict(inp)
        pred_label = labels[np.argmax(pred)]
        if pred_label == "Windows Desktop":
            pred_label = "Desktop"
        if pred_label == real_val:
            correct += 1
        # and print it for the humons
        # print(userAgent)
        print("%s : %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
        results_writer.writerow([x,labels[np.argmax(pred)]])

    print('Model Neural net - Accuracy on test data = {0:.4f}'.format(correct / cnt))