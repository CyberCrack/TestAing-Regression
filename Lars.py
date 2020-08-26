from sklearn.linear_model import Lars
from sklearn.model_selection import GridSearchCV

from Utility import printMetrics, getMetrics, logSave, saveBestParams, getPlantsPropulsionData


def LarsRegressor(X_train, X_test, y_train, y_test):
	reg = Lars()
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)

	printMetrics(y_true=y_test, y_pred=y_pred)
	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = reg.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="LarsRegressor", reg=reg, metrics=metrics, val_metrics=val_metrics)


def LarsRegressorGS(X_train, X_test, y_train, y_test):
	reg = Lars()
	grid_values = {
		'n_nonzero_coefs': list(range(100, 500, 100)),
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

X_train, X_test, y_train, y_test = getPlantsPropulsionData(splitData=True, makePolynomialFeatures=True)
LarsRegressor(X_train, X_test, y_train, y_test)