from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import sys
import json
cls = joblib.load('model.h5')
userAgent = sys.argv[1]
vec = joblib.load('vec_count.joblib')
features = vec.transform(
    [userAgent.lower()]
)
features_userAgent = features.toarray()
print(features_userAgent)
print(cls.predict(features_userAgent))
