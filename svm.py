#!usr/bin/env python\
import numpy as np
from sklearn import datasets
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler , PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR


class classification:
    def __init__(self):

        iris = datasets.load_iris()
        self.X = iris['data'][:,(2,3)]
        self.y = (iris['target'] == 2).astype(np.float64)

    def soft_margin(self):

            svm_clf =Pipeline((('scaler',StandardScaler()),
                                        ('linear_svc', LinearSVC(C = 1, loss ='hinge')),
                            ))
            svm_clf.fit(self.X,self.y)
            X_predict = svm_clf.predict([[5.5, 1.7]])
            print X_predict

    def pol_reg(self):
        polynomial_svm_clf = Pipeline((
            ('poly_features', PolynomialFeatures(degree = 3)),
                            ('scaler', StandardScaler()),
                            ('linear_svc', LinearSVC(C = 10, loss = 'hinge')),
                            ))
        polynomial_svm_clf.fit(self.X, self.y)
        X_predict = polynomial_svm_clf.predict([[5.5, 1.7]])
        print X_predict

    def pol_ker(self):
            polynomial_svm_clf = Pipeline((
                ('scaler', StandardScaler()),
                ('svm_clf', SVC(kernel = 'poly' , degree = 3, coef0 = 1, C = 5)),
            ))
            polynomial_svm_clf.fit(self.X, self.y)
            X_predict = polynomial_svm_clf.predict([[5.5, 1.7]])
            print X_predict
    def rbf_kern(self):
            rbf_kernel_svm_clf = Pipeline((
                ('scaler', StandardScaler()),
                ('svm_clf', SVC(kernel = 'rbf' , gamma = 10,  C = 10)),
            ))
            rbf_kernel_svm_clf.fit(self.X, self.y)
            X_predict =rbf_kernel_svm_clf.predict([[5.5, 1.7]])
            print X_predict

    def svm_reg(self):
        svm_reg =  LinearSVR(epsilon = 1.5)
        svm_reg.fit(self.X, self.y)
        X_predict = svm_reg.predict([[5.5, 1.7]])
        print X_predict
    def svm_reg_pol(self):
        svm_reg_pol =  SVR(kernel = 'poly', degree =2, C =100, epsilon = 100)
        svm_reg_pol.fit(self.X, self.y)
        X_predict = svm_reg_pol.predict([[5.5, 1.7]])
        print X_predict

if __name__ == '__main__':
        classification().svm_reg_pol()