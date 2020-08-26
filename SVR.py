import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def SVRRegressor(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	reg1 = SVR(kernel='poly')
	reg1.fit(X_train, y_train1)
	reg2 = SVR(kernel='poly')
	reg2.fit(X_train, y_train2)
	y_pred1 = reg1.predict(X=X_test)
	y_pred2 = reg2.predict(X=X_test)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred1 = reg1.predict(X=X_train)
	y_pred2 = reg2.predict(X=X_train)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="SVRRegressor", reg=[reg1, reg2], metrics=metrics, val_metrics=val_metrics)


def SVRRegressorGS(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	reg1 = SVR()
	reg2 = SVR()
	grid_values = {
		'kernel': ['poly', 'rbf'],
		'degree': list(range(1, 3)),
		'C': [value * 0.1 for value in range(0, 3)]
	}

	grid_reg1 = GridSearchCV(reg1, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							 refit='r2',
							 n_jobs=-1, cv=2, verbose=100)
	grid_reg1.fit(X_train, y_train1)
	reg1 = grid_reg1.best_estimator_
	reg1.fit(X_train, y_train1)
	grid_reg2 = GridSearchCV(reg2, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							 refit='r2',
							 n_jobs=-1, cv=2, verbose=100)
	grid_reg2.fit(X_train, y_train2)
	reg2 = grid_reg1.best_estimator_
	reg2.fit(X_train, y_train2)
	y_pred1 = reg1.predict(X=X_test)
	y_pred2 = reg2.predict(X=X_test)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred1 = reg1.predict(X=X_train)
	y_pred2 = reg2.predict(X=X_train)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	best_params1: dict = grid_reg1.best_params_
	best_params2: dict = grid_reg2.best_params_
	best_params = {}
	for key in best_params1.keys():
		best_params[key] = [best_params1[key], best_params2[key]]
	saveBestParams(nameOfModel="SVRRegressorGS", best_params=best_params)
	logSave(nameOfModel="SVRRegressorGS", reg=[reg1, reg2], metrics=metrics, val_metrics=val_metrics)
