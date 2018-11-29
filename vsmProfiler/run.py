from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas
import numpy
import sys
df = pandas.read_csv("browscap.csv", delimiter=",",dtype="string")
mask = ((df['PropertyName'].str.len() > 30) & ( (df['Device_Type'].str.contains('Mobile Phone')) | (df['Device_Type'].str.contains('Desktop')) | (df['Device_Type'].str.contains('Tablet')) | (df['Device_Type'].str.contains('Ebook Reader')) | (df['Device_Type'].str.contains('TV Device'))))
df = df.loc[mask]

df = df[df.Device_Type != '0']
df = df[df.Device_Type != '']
df = df.reindex()
newDF = pandas.DataFrame() 
df = newDF.append(df, ignore_index = True) # ignoring index is optional
from sklearn.cross_validation import train_test_split
docs = list(df['PropertyName'])
data_labels = list(df['Device_Type'])

X_train, X_test, y_train, y_test  = train_test_split(
        docs,
        data_labels,
        train_size=0.80,
        random_state=1234)

docs = list(X_train)
tf = TfidfVectorizer(analyzer='word')
tfidf_matrix =  tf.fit_transform(docs)
print "Result:"
i = 0
right = 0
wrong = 0
for agent in X_test:
    q_matrix = tf.transform([agent])
    result = cosine_similarity(q_matrix, tfidf_matrix)
    r = numpy.argmax(result)
    if (y_train[r] == y_test[i]):
        right+=1
    else:
        wrong+=1
    i+=1
    if(right!=0): print( "Accuracy:",right/((wrong+right)*1.0),right,wrong)
