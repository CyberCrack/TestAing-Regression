import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def PoissonReg(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	clf1 = PoissonRegressor()
	clf1.fit(X_train, y_train1)
	clf2 = PoissonRegressor()
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

	logSave(nameOfModel="PoissonReg", clf=[clf1, clf2], metrics=metrics, val_metrics=val_metrics)


def PoissonRegGS(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	clf1 = PoissonRegressor()
	clf2 = PoissonRegressor()
	grid_values = {
		'alpha': list(range(1, 11))
	}

	grid_clf1 = GridSearchCV(clf1, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							 refit='r2',
							 n_jobs=4, cv=5, verbose=100)
	grid_clf1.fit(X_train, y_train1)
	clf1 = grid_clf1.best_estimator_
	clf1.fit(X_train, y_train1)
	grid_clf2 = GridSearchCV(clf2, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							 refit='r2',
							 n_jobs=4, cv=5, verbose=100)
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
	saveBestParams(nameOfModel="PoissonRegGS", best_params=best_params)
	logSave(nameOfModel="PoissonRegGS", clf=[clf1, clf2], metrics=metrics, val_metrics=val_metrics)
