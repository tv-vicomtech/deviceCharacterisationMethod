from sklearn.externals import joblib
import csv
cls = joblib.load('model.h5')



fo = open("userAgents5.csv", "r")
lines=fo.readlines()
vec = joblib.load('vec_count.joblib')
with open('results_logRegfinal.csv', mode='w') as results_file:
	results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	correct = 0
	cnt = 0
	for x in range(1,len(lines)):
		cnt += 1
		if x <= 1000:
			real_val = 'Mobile Phone'
		elif x <= 2000:
			real_val = 'Tablet'
		elif x <= 3000:
			real_val = 'Desktop'
		else:
			real_val = 'TV Device'

		userAgent = lines[x]		
		features = vec.transform(
		    [userAgent.lower()]
		)
		features_userAgent = features.toarray()
		pred = cls.predict(features_userAgent)[0]
		results_writer.writerow([x, pred])
		if real_val == pred:
			correct += 1
		#print(cls.predict(features_userAgent))

	print('Model Logistic Regression - Accuracy on test data = {0:.4f}'.format(correct / cnt))