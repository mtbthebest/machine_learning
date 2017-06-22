#!usr/bin/env python
import numpy as np
from sklearn.linear_model import  SGDRegressor
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)


sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
print sgd_reg.intercept_, sgd_reg.coef_