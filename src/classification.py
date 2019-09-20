from numpy import load, expand_dims
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from random import choice

import warnings
warnings.filterwarnings('ignore')

data = load('../data/faces-embeddings.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = accuracy_score(trainY, yhat_train)
score_test = accuracy_score(testY, yhat_test)

data = load('../data/faces-dataset.npz')
testX_faces = data['arr_2']
selection = choice([i for i in range(testX.shape[0])])

random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testY[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])