from sklearn.linear_model import Lars
from sklearn.model_selection import GridSearchCV

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def LarsRegressor(X_train, X_test, y_train, y_test):
	clf = Lars()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	printMetrics(y_true=y_test, y_pred=y_pred)
	# Okaish
	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = clf.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="LarsRegressor", clf=clf, metrics=metrics, val_metrics=val_metrics)


def LarsRegressorGS(X_train, X_test, y_train, y_test):
	clf = Lars()
	grid_values = {
		'n_nonzero_coefs': list(range(100, 1001, 100)),
	}
	grid_clf = GridSearchCV(clf, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							refit='r2',
							n_jobs=4, cv=5, verbose=100)
	grid_clf.fit(X_train, y_train)
	clf = grid_clf.best_estimator_
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = clf.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	best_params: dict = grid_clf.best_params_
	saveBestParams(nameOfModel="LarsRegressorGS", best_params=best_params)
	logSave(nameOfModel="LarsRegressorGS", clf=clf, metrics=metrics, val_metrics=val_metrics)
