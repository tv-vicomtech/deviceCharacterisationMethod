import pandas as pd
import xgboost
from sklearn import preprocessing, naive_bayes, linear_model, ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from pathlib import Path
from utility_functions import train_model

# define filenames used to store model and data input
enc_labels = 'encoded_labels.joblib'
feat_store = 'features.joblib'
count_vec = 'vec_count.joblib'
split_data = 'split_data.joblib'

df = pd.read_csv("browscap.csv", delimiter=",", dtype="str")
mask = ((df['PropertyName'].str.len() > 30) &
        ((df['Device_Type'].str.contains('Mobile Phone')) |
         (df['Device_Type'].str.contains('Desktop')) |
         (df['Device_Type'].str.contains('Tablet')) |
         (df['Device_Type'].str.contains('Ebook Reader')) |
         (df['Device_Type'].str.contains('TV Device'))))
df = df.loc[mask]
df = df[df.Device_Type != '0']
df = df[df.Device_Type != '']
df = df[:180000]
df = df.reindex()
newDF = pd.DataFrame()
df = newDF.append(df, ignore_index=True)
df = df[['PropertyName', 'Device_Type']]
data = df['PropertyName']
data_labels = df['Device_Type']
print(df.head())
print('----------------------')
print(df.columns.values.tolist())
print('----------------------')
print(df.groupby('Device_Type').describe())

# If feature data available, load it
if Path(feat_store).is_file() and Path(count_vec).is_file():
    features_nd = joblib.load(feat_store)
    count_vec = joblib.load(count_vec)
else:  # compute it
    count_vec = CountVectorizer(analyzer='word', lowercase=True,)
    features = count_vec.fit_transform(data)
    features_nd = features.toarray()
    joblib.dump(count_vec, count_vec)
    joblib.dump(features_nd, feat_store)

if Path(split_data).is_file():
    split_data = joblib.load(split_data)
    [X_train, X_test, y_train, y_test] = split_data
else:
    print('Splitting dataset')
    X_train, X_test, y_train, y_test = train_test_split(features_nd, data_labels, test_size=0.20, random_state=1234)
    split_data = [X_train, X_test, y_train, y_test]
    joblib.dump(split_data, split_data)


print('Encoding Labels')
# label encode the target variable
if Path(enc_labels).is_file():
    encoder = joblib.load(enc_labels)
    print(encoder.classes_)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
else:
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(data_labels)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    joblib.dump(encoder, enc_labels)

print('PREPROCESSING DONE - NOW RUNNING CLASSIFIERS')

## NAIVE BAYES ##########
accuracy = train_model(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test,
                       model_name='naive_bayes.h5')
print('NAIVE BAYES - Accuracy:  {0:.2f}'.format(accuracy))
#########################

## LINEAR CLASSIFIER ####
accuracy = train_model(linear_model.LogisticRegression(),
                       X_train, y_train, X_test, y_test, model_name='linear_default.h5')
print('LINEAR CLASSIFIER - Accuracy:  {0:.2f}'.format(accuracy))
#########################

## RANDOM FORESTS #######
for trees in range (10, 101, 10):
    name = 'random_forest_default_depth_estimators_' + str(trees) + '.h5'
    accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=trees, random_state=1234),
                           X_train, y_train, X_test, y_test, model_name=name)
    print('RANDOM FOREST ({0} trees, deafult depth - Accuracy:  {1:.2f}'.format(trees,accuracy))
#########################

## XGBOOST ##############
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['learning_rate'] = 0.01
accuracy = train_model(xgboost.XGBClassifier(**param),
                       X_train, y_train, X_test, y_test, model_name='XGBoost.h5')
print('XGBOOST - Accuracy: {0:.2f}'.format(accuracy))
#########################
