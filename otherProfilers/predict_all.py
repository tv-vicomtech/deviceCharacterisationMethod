from sklearn.externals import joblib
import os
import csv

# load features
vec = joblib.load('vec_count.joblib')

# load label encoders
encoder = joblib.load('encoded_labels.joblib')

# predict for all the models in the folder
for file in [f for f in os.listdir('./') if f.endswith('.h5')]:
    model = joblib.load(file)
    out_pred_name = 'results_' + file + '.csv'

    with open(out_pred_name, 'w') as results_file, open("userAgents5.csv", "r") as test_data:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cnt = 1
        correct = 0
        for userAgent in test_data:
            features = vec.transform([userAgent.lower()])
            features_userAgent = features.toarray()
            prediction = model.predict(features_userAgent)
            prediction_readable = encoder.inverse_transform(prediction)
            where_are_we = (cnt-2) // 1000
            if where_are_we == 0:
                real_value = 'Mobile Phone'
            elif where_are_we == 1:
                real_value = 'Tablet'
            elif where_are_we == 2:
                real_value = 'Desktop'
            else:
                real_value = 'TV Device'
            if prediction_readable == real_value:
                correct += 1
            results_writer.writerow([cnt,prediction_readable])
            cnt += 1

    print('Model {0} - Accuracy on test data = {1:.2f}'.format(file, correct / cnt))