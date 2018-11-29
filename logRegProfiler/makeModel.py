from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pandas
import json
data = []
data_labels = []
df = pandas.read_csv("browscap.csv", delimiter=",",dtype="string")
mask = ((df['PropertyName'].str.len() > 30) & ( (df['Device_Type'].str.contains('Mobile Phone')) | (df['Device_Type'].str.contains('Desktop')) | (df['Device_Type'].str.contains('Tablet')) | (df['Device_Type'].str.contains('Ebook Reader')) | (df['Device_Type'].str.contains('TV Device'))))
df = df.loc[mask]
df = df[df.Device_Type != '0']
df = df[df.Device_Type != '']
df = df [:180000]
df = df.reindex()
newDF = pandas.DataFrame() 
df = newDF.append(df, ignore_index = True)
data = df['PropertyName']
data_labels = df['Device_Type'];
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = True,
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray()
joblib.dump(vectorizer, 'vec_count.joblib')
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
from sklearn.externals import joblib
joblib.dump(log_model, 'model.h5') 
y_pred = log_model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
