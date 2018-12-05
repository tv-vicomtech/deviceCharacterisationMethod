from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pandas
import csv
from pathlib import Path

path_browscap = Path(os.getcwd()).parent / "UA" / "browscap.csv"
df = pandas.read_csv(str(path_browscap), delimiter=",", dtype="str")
mask = ((df['PropertyName'].str.len() > 30) &
        ((df['Device_Type'].str.contains('Mobile Phone')) |
         (df['Device_Type'].str.contains('Desktop')) |
         (df['Device_Type'].str.contains('Tablet')) |
         (df['Device_Type'].str.contains('TV Device'))))
df = df.loc[mask]
df = df[df.Device_Type != '0']
df = df[df.Device_Type != '']
# df = df[:180000]
df = df.reindex()
newDF = pandas.DataFrame() 
df = newDF.append(df, ignore_index=True)
data = df['PropertyName']
vectorizer = CountVectorizer(analyzer='word', lowercase=True)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray()


data_labels = df['Device_Type']

joblib.dump(vectorizer, 'vec_count.joblib')

X_train, X_test, y_train, y_test = train_test_split(features_nd, data_labels, test_size=0.20, random_state=1234)

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)  # type: LogisticRegression

joblib.dump(log_model, 'model.h5') 
y_pred = log_model.predict(X_test)

print(accuracy_score(y_test, y_pred))
