from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def LassoRegressor(X_train, X_test, y_train, y_test):
	reg = Lasso(alpha=0.01)
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = reg.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="LassoRegressor", reg=reg, metrics=metrics, val_metrics=val_metrics)


def LassoRegressorGS(X_train, X_test, y_train, y_test):
	reg = Lasso()
	grid_values = {
		'alpha': list(range(1, 3)) + [value * 0.01 for value in range(1, 3)],
	}
	grid_reg = GridSearchCV(reg, param_grid=grid_values, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
							refit='r2',
							n_jobs=-1, cv=2, verbose=100)
	grid_reg.fit(X_train, y_train)
	reg = grid_reg.best_estimator_
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = reg.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	best_params: dict = grid_reg.best_params_
	saveBestParams(nameOfModel="LarsRegressorGS", best_params=best_params)
	logSave(nameOfModel="LarsRegressorGS", reg=reg, metrics=metrics, val_metrics=val_metrics)
