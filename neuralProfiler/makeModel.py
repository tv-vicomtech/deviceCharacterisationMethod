import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn.externals import joblib
import time
# extract data from a csv
# notice the cool options to skip lines at the beginning
# and to only take data from certain columns
t1 = time.time()
df = pd.read_csv("browscap.csv", delimiter=",",dtype="string")
print df['Device_Type'].value_counts()

mask = ((df['PropertyName'].str.len() > 30) & ( (df['Device_Type'].str.contains('Mobile Phone')) | (df['Device_Type'].str.contains('Desktop')) | (df['Device_Type'].str.contains('Tablet')) | (df['Device_Type'].str.contains('Ebook Reader')) | (df['Device_Type'].str.contains('TV Device'))))
df = df.loc[mask]
df = df[df.Device_Type != '0']
df = df[df.Device_Type != '']
df = df[df.Device_Type.notna() ]
df = shuffle(df)

docs = df['PropertyName']
# create our training data from the tweets
train_x = docs.tolist()
# index all the sentiment labels
train_y = df['Device_Type'].tolist()
# only work with the 3000 most popular words found in our dataset
max_words = 4090
# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)
# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

# Let's save this out so we can use it later

def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit(train_y)
#print(integer_encoded)
trainY = label_encoder.transform(train_y)
trainY = keras.utils.to_categorical(trainY,8)
joblib.dump(label_encoder,"laberEncoder.h5")
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(128, input_shape=(4090,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(train_x, trainY,
    batch_size=32,
    epochs=5,
    verbose=1,
    validation_split=0.1,
    shuffle=True)


model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
t2 = time.time()
print ("time:",t2-t1)
model.save_weights('model.h5')
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
