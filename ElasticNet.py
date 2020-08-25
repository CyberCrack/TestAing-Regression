from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def ElasticNetRegressor(X_train, X_test, y_train, y_test):
	clf = ElasticNet(alpha=10, l1_ratio=0.2)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = clf.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="ElasticNetRegressor", clf=clf, metrics=metrics, val_metrics=val_metrics)


# AVOID

def ElasticNetRegressorGS(X_train, X_test, y_train, y_test):
	clf = ElasticNet()
	grid_values = {
		'alpha': list(range(1, 5)) + [value * 0.01 for value in range(1, 10)],
		'l1_ratio': [0, 0.5, 1]

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
	saveBestParams(nameOfModel="ElasticNetRegressorGS", best_params=best_params)
	logSave(nameOfModel="ElasticNetRegressorGS", clf=clf, metrics=metrics, val_metrics=val_metrics)
