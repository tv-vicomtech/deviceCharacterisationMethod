from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import pandas
import numpy
import csv

path_browscap = Path(os.getcwd()).parent / "UA" / "browscap.csv"
df = pandas.read_csv(str(path_browscap), delimiter=",", dtype="str")
mask = ((df['PropertyName'].str.len() > 30) &
        ((df['Device_Type'].str.contains('Mobile Phone')) |
         (df['Device_Type'].str.contains('Desktop')) |
         (df['Device_Type'].str.contains('Tablet')) |
         (df['Device_Type'].str.contains('Ebook Reader')) |
         (df['Device_Type'].str.contains('TV Device'))))
df = df.loc[mask]
df = df[df.Device_Type != '0']
df = df[df.Device_Type != '']
df = df.reindex()
newDF = pandas.DataFrame() 
df = newDF.append(df, ignore_index=True)  # ignoring index is optional

fo = open("userAgents5.csv", "r")
lines = fo.readlines()

docs = list(df['PropertyName'])
tf = TfidfVectorizer(analyzer='word')
tfidf_matrix = tf.fit_transform(docs)
print("Result:")

with open('results_vsm5.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for x in range(1, len(lines)):
        q_matrix = tf.transform([lines[x]])
        result = cosine_similarity(q_matrix, tfidf_matrix)
        r = numpy.argmax(result)
        print(x, df['Device_Type'][r])
        results_writer.writerow([x, df['Device_Type'][r]])
