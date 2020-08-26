import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def NuSVRRegressor(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	clf1 = NuSVR()
	clf1.fit(X_train, y_train1)
	clf2 = NuSVR()
	clf2.fit(X_train, y_train2)
	y_pred1 = clf1.predict(X=X_test)
	y_pred2 = clf2.predict(X=X_test)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred1 = clf1.predict(X=X_train)
	y_pred2 = clf2.predict(X=X_train)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="NuSVRRegressor", clf=[clf1, clf2], metrics=metrics, val_metrics=val_metrics)


def NuSVRRegressorGS(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	clf1 = NuSVR()
	clf2 = NuSVR()
	grid_values = {
		'nu': [value * 0.1 for value in range(1, 3)], 'C': list(range(1, 3)),
		'kernel': ['poly', 'rbf'], 'degree': list(range(1, 3))
	}

	grid_clf1 = GridSearchCV(clf1, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							 refit='r2',
							 n_jobs=-1, cv=2, verbose=100)
	grid_clf1.fit(X_train, y_train1)
	clf1 = grid_clf1.best_estimator_
	clf1.fit(X_train, y_train1)
	grid_clf2 = GridSearchCV(clf2, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							 refit='r2',
							 n_jobs=-1, cv=2, verbose=100)
	grid_clf2.fit(X_train, y_train2)
	clf2 = grid_clf1.best_estimator_
	clf2.fit(X_train, y_train2)
	y_pred1 = clf1.predict(X=X_test)
	y_pred2 = clf2.predict(X=X_test)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred1 = clf1.predict(X=X_train)
	y_pred2 = clf2.predict(X=X_train)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	best_params1: dict = grid_clf1.best_params_
	best_params2: dict = grid_clf2.best_params_
	best_params = {}
	for key in best_params1.keys():
		best_params[key] = [best_params1[key], best_params2[key]]
	saveBestParams(nameOfModel="NuSVRRegressorGS", best_params=best_params)
	logSave(nameOfModel="NuSVRRegressorGS", clf=[clf1, clf2], metrics=metrics, val_metrics=val_metrics)
