from sklearn import metrics
from sklearn.externals import joblib
from pathlib import Path

def train_model(classifier, feature_vector_train, label, feature_vector_valid, y_valid,
                is_neural_net=False, model_name=None):
    if Path(model_name).is_file():
        print('Accuracy already computed for model ' + model_name)
        return 0
    else:
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        if model_name:
            joblib.dump(classifier, model_name)

        return metrics.accuracy_score(predictions, y_valid)
