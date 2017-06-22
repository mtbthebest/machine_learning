#!usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from tensorflow.contrib import   batching

class Tree:
    def __init__(self):
        self.iris = load_iris()
        self.X =self. iris.data[:,2:]
        self.y=self.iris.target
    def tree_class(self):
        tree_clf = DecisionTreeClassifier(max_depth = 2)
        tree_clf.fit(self.X, self.y)
        graph_viz = export_graphviz(decision_tree = tree_clf,
                        out_file=("iris_tree.dot"),
                        feature_names=self.iris.feature_names[2:],
                        class_names=self.iris.target_names,
                        rounded=True,
                        filled=True)

        predict_proba = tree_clf.predict_proba([[5, 1.5]])
        predict = tree_clf.predict([[5, 1.5]])
        print predict
    def tree_reg(self):
        tree_reg = DecisionTreeRegressor(max_depth = 3)
        tree_reg.fit(self.X, self.y)
        graph_viz = export_graphviz(decision_tree = tree_reg,
                        out_file=("iris_tree_reg.dot"),
                        feature_names=self.iris.feature_names[2:],
                        class_names=self.iris.target_names,
                        rounded=True,
                        filled=True)



if __name__ == '__main__':
    Tree().tree_reg()