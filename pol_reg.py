#!usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
poly_features = PolynomialFeatures(degree = 2, include_bias = False)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
X_poly = poly_features.fit_transform(X)
print  X[0]
print X_poly[0]
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print lin_reg.intercept_, lin_reg.coef_

# plt.plot(X,y,'b.')
# plt.axis([0,3,0,15])
# plt.show()