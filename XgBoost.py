import numpy as np
import xgboost as xg
from sklearn.model_selection import GridSearchCV

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def XgBoost(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	reg1 = xg.XGBRegressor(objective='reg:squarederror')
	reg1.fit(X=X_train, y=y_train1)

	reg2 = xg.XGBRegressor(objective='reg:squarederror')
	reg2.fit(X=X_train, y=y_train2)

	y_pred1 = reg1.predict(X_test)
	y_pred2 = reg2.predict(X_test)

	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred1 = reg1.predict(X_train)
	y_pred2 = reg2.predict(X_train)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)
	logSave(nameOfModel="XgBoost", reg=[reg1, reg2], metrics=metrics, val_metrics=val_metrics)


def XgBoostGS(X_train, X_test, y_train, y_test):
	y_train1 = y_train[:, 0]
	y_train2 = y_train[:, 1]
	reg1 = xg.XGBRegressor(objective='reg:squarederror')
	reg2 = xg.XGBRegressor(objective='reg:squarederror')
	grid_values = {'learning_rate': [x / 10 for x in range(1, 5)], 'max_depth': list(range(11, 15))}
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
	y_pred1 = reg1.predict(X_test)
	y_pred2 = reg2.predict(X_test)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred1 = reg1.predict(X_train)
	y_pred2 = reg2.predict(X_train)
	y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	best_params1: dict = grid_reg1.best_params_
	best_params2: dict = grid_reg2.best_params_
	best_params = {}
	for key in best_params1.keys():
		best_params[key] = [best_params1[key], best_params2[key]]
	saveBestParams(nameOfModel="XgBoostGS", best_params=best_params)
	logSave(nameOfModel="XgBoostGS", reg=[reg1, reg2], metrics=metrics, val_metrics=val_metrics)
