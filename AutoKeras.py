import autokeras as ak

from Utility import getPlantsPropulsionData, getMetrics, printMetrics, logSave

X_train, X_test, y_train, y_test = getPlantsPropulsionData(splitData=True, makePolynomialFeatures=True)

reg = ak.StructuredDataRegressor(loss='mean_absolute_error', metrics=['mean_squared_error', 'mean_absolute_error'], objective='val_mean_absolute_error',
								 overwrite=True,
								 max_trials=10)
reg.fit(x=X_train, y=y_train, epochs=20, validation_data=(X_test, y_test))

y_preds = reg.predict(X_train)
printMetrics(y_true=y_train, y_pred=y_preds)
metrics = getMetrics(y_true=y_train, y_pred=y_preds)

y_preds = reg.predict(X_test)
printMetrics(y_true=y_test, y_pred=y_preds)
val_metrics = getMetrics(y_true=y_test, y_pred=y_preds)

logSave(nameOfModel="AutoKeras", reg=None, metrics=metrics, val_metrics=val_metrics)
