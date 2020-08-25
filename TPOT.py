import numpy as np
from tpot import TPOTRegressor

from Utility import getPlantsPropulsionData, printMetrics, getMetrics

X_train, X_test, y_train, y_test = getPlantsPropulsionData(splitData=True, makePolynomialFeatures=False)
y_train1 = y_train[:, 0]
y_train2 = y_train[:, 1]

tpotReg1 = TPOTRegressor(generations=50, population_size=50, max_time_mins=15, scoring='r2', verbosity=3, n_jobs=4)

tpotReg2 = TPOTRegressor(generations=50, population_size=50, max_time_mins=15, scoring='r2', verbosity=3, n_jobs=4)

tpotReg1.fit(X_train, y_train1)
tpotReg2.fit(X_train, y_train2)

y_pred1 = tpotReg1.predict(X_test)
y_pred2 = tpotReg2.predict(X_test)
y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

printMetrics(y_true=y_test, y_pred=y_pred)
val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)

y_pred1 = tpotReg1.predict(X_train)
y_pred2 = tpotReg1.predict(X_train)
y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

printMetrics(y_true=y_train, y_pred=y_pred)
metrics = getMetrics(y_true=y_train, y_pred=y_pred)

tpotReg1.export('tpot_pipeline.py')
