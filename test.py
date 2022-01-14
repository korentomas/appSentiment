import joblib
classifier = joblib.load('sentiment-model.pkl')
result = classifier.predict((['i love mandarinas']))
print(result)

