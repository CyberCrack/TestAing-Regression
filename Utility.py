import datetime
import os
import pickle

import h2o.automl
import pandas as pd
import xgboost as xg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class Variables(object):
	def __init__(self, pipeline: Pipeline):
		self.modelNum = str(abs(hash(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))
		self.pipeline = pipeline


Var: Variables


def getPlantsPropulsionData(splitData=True, makePolynomialFeatures=False):
	global Var
	data = pd.read_csv(filepath_or_buffer="Data Source/data.txt", sep="   ", header=None, engine='python')
	col_heading = ['Lever_position', 'Ship_speed', 'Gas_Turbine_shaft_torque', 'Gas_Turbine_rate_of_revolutions',
				   'Gas_Generator_rate_of_revolutions', 'Starboard_Propeller_Torque', 'Port_Propeller_Torque', 'HP_Turbine_exit_temperature',
				   'GT_Compressor_inlet_air_temperature', 'GT_Compressor_outlet_air_temperature', 'HP_Turbine_exit_pressure',
				   'GT_Compressor_inlet_air_pressure', 'GT_Compressor_outlet_air_pressure', 'Gas_Turbine_exhaust_gas_pressure',
				   'Turbine_Injecton_Control', 'Fuel_flow', 'GT_Compressor_decay_state_coefficient', 'GT_Turbine_decay_state_coefficient']
	col_to_drop = ['Lever_position', 'Ship_speed', 'GT_Compressor_inlet_air_temperature', 'GT_Compressor_inlet_air_pressure']
	data.columns = list(col_heading)
	data = data.drop(col_to_drop, axis=1)
	X = data.drop(['GT_Compressor_decay_state_coefficient', 'GT_Turbine_decay_state_coefficient'], axis=1).values
	y = data[col_heading[-2:]].values
	steps = [('scaler', StandardScaler())]
	if makePolynomialFeatures:
		steps.insert(0, ('polynomialfeatures', PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)))
	pipeline = Pipeline(steps=steps)
	Var = Variables(pipeline=pipeline)
	if splitData:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
		X_train = pipeline.fit_transform(X=X_train)
		X_test = pipeline.transform(X=X_test)
		return X_train, X_test, y_train, y_test
	else:
		X = pipeline.fit_transform(X=X)
		return X, y


def printMetrics(y_true, y_pred):
	mean_absolute_error_score = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
	print("mean_absolute_error:", mean_absolute_error_score)
	mean_squared_error_score = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
	print("mean_squared_error:", mean_squared_error_score)
	r2_score_error = r2_score(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
	print("r2_score:", r2_score_error)


def getMetrics(y_true, y_pred):
	mean_absolute_error_score = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
	mean_squared_error_score = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
	r2_score_error = r2_score(y_true=y_true, y_pred=y_pred, multioutput='uniform_average')
	return mean_absolute_error_score, mean_squared_error_score, r2_score_error


def logSave(nameOfModel, reg, metrics, val_metrics):
	mean_absolute_error_score, mean_squared_error_score, r2_score_error = metrics
	val_mean_absolute_error_score, val_mean_squared_error_score, val_r2_score_error = val_metrics
	msg = str(Var.modelNum) + "-" + nameOfModel + "\t\t" + "mae-" + str(mean_absolute_error_score) + "\tmse-" + str(
		mean_squared_error_score) + "\tr2-" + str(r2_score_error) + "\tval_mae-" + str(val_mean_absolute_error_score) + "\tval_mse-" + str(
		val_mean_squared_error_score) + "\tval_r2-" + str(val_r2_score_error) + "\n"
	f = open("SKlogs.log", "a+")
	f.write(msg)
	f.close()
	if not os.path.exists("SKMetrics.csv"):
		f = open("SKMetrics.csv", "w")
		f.write(",".join(
			["Model No.", "Model Type", "mean_absolute_error", "mean_squared_error", "r2_score", "val_mean_absolute_error", "val_mean_squared_error",
			 "val_r2_score"]) + "\n")
		f.close()
	f = open("SKMetrics.csv", "a+")
	msg = ",".join(
		[Var.modelNum, nameOfModel, str(mean_absolute_error_score), str(mean_squared_error_score), str(r2_score_error),
		 str(val_mean_absolute_error_score), str(val_mean_squared_error_score), str(val_r2_score_error)
		 ])
	f.write(msg + "\n")
	f.close()
	if not os.path.exists("DataPreprocessingPipeline"):
		os.mkdir("DataPreprocessingPipeline")
	name_of_file = "_".join([Var.modelNum, nameOfModel, "DataPreprocessingPipeline"]) + ".pickle"
	pickle_out = open(os.path.join("DataPreprocessingPipeline", name_of_file), "wb")
	pickle.dump(Var.pipeline, pickle_out)
	if not os.path.exists("SKLearnModels"):
		os.mkdir("SKLearnModels")
	if not os.path.exists("H2OModels"):
		os.mkdir("H2OModels")
	if reg is None:
		return
	if isinstance(reg, list):
		if "H2O" in nameOfModel:
			name_of_file = "_".join([Var.modelNum, nameOfModel])
			h2o.save_model(reg[0].leader, path=os.path.join("H2OModels", name_of_file + "1"))
			h2o.save_model(reg[1].leader, path=os.path.join("H2OModels", name_of_file + "2"))
		elif type(reg) is xg.XGBRegressor:
			name_of_file = "_".join([Var.modelNum, nameOfModel]) + ".bin"
			reg[0].save_model(os.path.join("SKLearnModels", name_of_file + "1"))
			reg[1].save_model(os.path.join("SKLearnModels", name_of_file + "2"))
		else:
			name_of_file = "_".join([Var.modelNum, nameOfModel]) + ".pickle"
			pickle_out = open(os.path.join("SKLearnModels", name_of_file + "1"), "wb")
			pickle.dump(reg[0], pickle_out)
			pickle_out = open(os.path.join("SKLearnModels", name_of_file + "2"), "wb")
			pickle.dump(reg[1], pickle_out)
	else:
		name_of_file = "_".join([Var.modelNum, nameOfModel]) + ".pickle"
		pickle_out = open(os.path.join("SKLearnModels", name_of_file), "wb")
		pickle.dump(reg, pickle_out)


def saveBestParams(nameOfModel, best_params):
	f = open("GridSearchParams.txt", "a+")
	f.write(Var.modelNum + "-" + nameOfModel + "\t" + str(best_params) + "\n")
	f.close()
