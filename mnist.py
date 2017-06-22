#!usr/bin/env python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score,recall_score,precision_recall_curve


mnist = fetch_mldata('MNIST original')
print mnist
X,y = mnist['data'],mnist  ['target']
print X.shape
print y.shape
some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest')
plt.axis('off')
#plt.show()
print y[66000]
X_train, X_test,y_train, y_test = X[:60000], X[60000:],y[:60000],y[60000:]

shuffle_index = np.random.permutation(60000)
X_train,y_train = X_train[shuffle_index], y[shuffle_index]

y_train_5 = (y_train ==5)
y_test_5 =  (y_test == 5)
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train,y_train_5)
print sgd_clf.predict([some_digit])
print cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring = 'accuracy')
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv =3)
print confusion_matrix(y_train_5,y_train_pred)
print precision_score(y_train_5,y_train_pred)
print recall_score(y_train_5,y_train_pred)

y_scores = sgd_clf.decision_function([some_digit])
print y_scores
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print y_some_digit_pred
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")
forest_clf  = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")