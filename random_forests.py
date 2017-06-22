#!usr/bin/env python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier,GradientBoostingRegressor
from sklearn.svm import  SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.tree import  DecisionTreeClassifier, DecisionTreeRegressor
class RandomForest:
    def __init__(self):
        self.X , self.y= make_moons(n_samples = 500, noise = 0.3, random_state = 42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state = 42)

    def voting_class(self):
        log_clf = LogisticRegression()
        rnd_clf = RandomForestClassifier()
        svm_clf = SVC()
        voting_clf = VotingClassifier(estimators = [('lr', log_clf), ('rf', rnd_clf),('svc', svm_clf)],
                                      voting = 'hard')
        voting_clf.fit(self.X_train, self.y_train)

        for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            print clf.__class__.__name__, accuracy_score(self.y_test, y_pred)
    def bagging(self):
        bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, max_samples = 100, bootstrap = True, n_jobs = 1, oob_score = True)
        bag_clf.fit(self.X_train, self.y_train)
        y_pred = bag_clf.predict(self.X_test)
        print  accuracy_score(self.y_test, y_pred)
        print bag_clf.oob_decision_function_()
    def randomforest(self):
        rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, n_jobs = -1)
        rnd_clf.fit(self.X_train, self.y_train)
        y_pred_rf = rnd_clf.predict(self .X_train)
        print  y_pred_rf
    def ada_clf(self):
        ada_clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth = 1), n_estimators = 200,
            algorithm = "SAMME.R", learning_rate = 0.5
        )
        ada_clf.fit(self.X_train, self.y_train)
        y_pred_rf = ada_clf.predict(self.X_train)
        print  y_pred_rf
    def grad_boost(self):
        gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators = 3, learning_rate = 1.0)
        gbrt.fit(self.X_train, self.y_train)
        y_pred_rf = gbrt.predict(self.X_train)
        print  y_pred_rf

if __name__ == '__main__':
    RandomForest().grad_boost()