from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from Utility import printMetrics, getMetrics, logSave, saveBestParams


def NeuralNet(X_train, X_test, y_train, y_test):
	clf = MLPRegressor(hidden_layer_sizes=(32, 64, 128, 256, 128, 64))
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	printMetrics(y_true=y_test, y_pred=y_pred)

	val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
	y_pred = clf.predict(X=X_train)
	metrics = getMetrics(y_true=y_train, y_pred=y_pred)

	printMetrics(y_true=y_train, y_pred=y_pred)

	logSave(nameOfModel="NeuralNet", clf=clf, metrics=metrics, val_metrics=val_metrics)


def NeuralNetGS(X_train, X_test, y_train, y_test):
	clf = MLPRegressor()
	grid_values = {
		'hidden_layer_sizes': [(8, 16, 32, 64, 128, 64, 32, 64, 16, 8), (8, 16, 32, 64, 32, 16, 8), (8, 16, 32, 16, 8)],
		'activation': ['tanh', 'relu'],
		'solver': ['lbfgs', 'sgd', 'adam'],
		'learning_rate': ['constant', 'invscaling']

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
	saveBestParams(nameOfModel="NeuralNetGS", best_params=best_params)
	logSave(nameOfModel="NeuralNetGS", clf=clf, metrics=metrics, val_metrics=val_metrics)
