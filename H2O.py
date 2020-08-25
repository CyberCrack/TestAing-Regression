import h2o
import numpy as np
from h2o.automl import H2OAutoML

from Utility import getPlantsPropulsionData, getMetrics, printMetrics, logSave

X_train, X_test, y_train, y_test = getPlantsPropulsionData(splitData=True, makePolynomialFeatures=True)
y_train1 = y_train[:, 0]
y_train2 = y_train[:, 1]
y_test1 = y_test[:, 0]
y_test2 = y_test[:, 1]

h2o.init(ignore_config=True)

trainFrame1 = h2o.H2OFrame(np.concatenate((X_train, y_train1.reshape(-1, 1)), axis=1))
testFrame1 = h2o.H2OFrame(np.concatenate((X_test, y_test1.reshape(-1, 1)), axis=1))
x_labels1 = list(trainFrame1.columns)
y_labels1 = x_labels1[-1]
x_labels1.remove(y_labels1)

trainFrame2 = h2o.H2OFrame(np.concatenate((X_train, y_train2.reshape(-1, 1)), axis=1))
testFrame2 = h2o.H2OFrame(np.concatenate((X_test, y_test2.reshape(-1, 1)), axis=1))
x_labels2 = list(trainFrame2.columns)
y_labels2 = x_labels2[-1]
x_labels2.remove(y_labels2)

aml1 = H2OAutoML(max_runtime_secs=120)
aml1.train(x=x_labels1, y=y_labels1, training_frame=trainFrame1, validation_frame=testFrame1)

aml2 = H2OAutoML(max_runtime_secs=120)
aml2.train(x=x_labels2, y=y_labels2, training_frame=trainFrame2, validation_frame=testFrame2)

y_predsFrame1 = aml1.leader.predict(testFrame1)
y_test_pred_df1 = y_predsFrame1.as_data_frame()
y_predsFrame1 = aml1.leader.predict(trainFrame1)
y_train_pred_df1 = y_predsFrame1.as_data_frame()

y_predsFrame2 = aml2.leader.predict(testFrame2)
y_test_pred_df2 = y_predsFrame2.as_data_frame()
y_predsFrame2 = aml2.leader.predict(trainFrame2)
y_train_pred_df2 = y_predsFrame2.as_data_frame()

y_preds1 = y_test_pred_df1['predict'].values.reshape(-1, 1)
y_preds2 = y_test_pred_df2['predict'].values.reshape(-1, 1)
y_pred = np.hstack((y_preds1, y_preds2))
val_metrics = getMetrics(y_true=y_test, y_pred=y_pred)
printMetrics(y_true=y_test, y_pred=y_pred)

y_preds1 = y_train_pred_df1['predict'].values.reshape(-1, 1)
y_preds2 = y_train_pred_df2['predict'].values.reshape(-1, 1)
y_pred = np.hstack((y_preds1, y_preds2))
metrics = getMetrics(y_true=y_train, y_pred=y_pred)
printMetrics(y_true=y_train, y_pred=y_pred)

# logSave(nameOfModel="H2O", clf=[aml1, aml2], metrics=metrics, val_metrics=val_metrics)
# h2o.shutdown(prompt=True)
