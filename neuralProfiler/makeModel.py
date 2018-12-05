import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from pathlib import Path
import os
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from collections import Counter

# extract data from a csv
# notice the cool options to skip lines at the beginning
# and to only take data from certain columns
path_browscap = Path(os.getcwd()).parent / "UA" / "browscap.csv"
df = pd.read_csv(str(path_browscap), delimiter=",", dtype="str")
mask = ((df['PropertyName'].str.len() > 30) &
        ((df['Device_Type'].str.contains('Mobile Phone')) |
         (df['Device_Type'].str.contains('Desktop')) |
         (df['Device_Type'].str.contains('Tablet')) |
         (df['Device_Type'].str.contains('TV Device'))))
df = df.loc[mask]
df = df[df.Device_Type != '0']
df = df[df.Device_Type != '']
df = df[df.Device_Type.notna()]
docs = df['PropertyName']

# create our training data from the tweets
X = docs.tolist()
#print(len(X))
# replace general desktop and window desktop as Desktop, and general mobile phone as Mobile Phone
X = [x.replace('Windows Desktop', 'Desktop') for x in X]
X = [x.replace('general Desktop', 'Desktop') for x in X]
X = [x.replace('general Mobile Phone', 'Mobile Phone') for x in X]

# index all the device labels
Y = df['Device_Type'].tolist()

Y = [x.replace('Windows Desktop', 'Desktop') for x in Y]
Y = [x.replace('general Desktop', 'Desktop') for x in Y]
Y = [x.replace('general Mobile Phone', 'Mobile Phone') for x in Y]
#print(set(Y))
#c = Counter(Y)
#print(c)

# only work with the most popular words found in our dataset
max_words = 3500 #4090
# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(X)
# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(my_text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(my_text)]


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in X:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
X = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit(Y)
print(label_encoder.classes_)
# print(integer_encoded)
Y = label_encoder.transform(Y)
Y = keras.utils.to_categorical(Y, 4)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.10, random_state=1234)

model = Sequential()
model.add(Dense(128, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(4))
model.add(Activation('softmax'))
model.summary()

callbacks_list = [
                  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1),
                  keras.callbacks.ModelCheckpoint(filepath='model_ste.h5', monitor='val_loss', save_best_only=True),
                  keras.callbacks.TensorBoard(log_dir='tf_logs', histogram_freq=1)
                 ]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=512,
                    epochs=15,
                    verbose=1,
                    validation_data=(X_val, Y_val),
                    callbacks=callbacks_list)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
