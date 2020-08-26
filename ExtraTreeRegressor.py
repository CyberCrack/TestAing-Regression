from sklearn.model_selection import GridSearchCV
from sklearn.tree import ExtraTreeRegressor

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def ExtraTree(X_train, X_test, y_train, y_test):
	clf = ExtraTreeRegressor()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = clf.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="ExtraTree", clf=clf, metrics=metrics, val_metrics=val_metrics)


def ExtraTreeGS(X_train, X_test, y_train, y_test):
	clf = ExtraTreeRegressor()
	grid_values = {
		'criterion': ["mse", "mae"],
		'max_depth': list(range(20, 25))
	}
	grid_clf = GridSearchCV(clf, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							refit='r2',
							n_jobs=-1, cv=2, verbose=100)
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
	saveBestParams(nameOfModel="ExtraTreeGS", best_params=best_params)
	logSave(nameOfModel="ExtraTreeGS", clf=clf, metrics=metrics, val_metrics=val_metrics)