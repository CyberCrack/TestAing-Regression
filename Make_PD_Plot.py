import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import plot_partial_dependence
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

from Utility import getMetrics, printMetrics

data = pd.read_csv(filepath_or_buffer="Data Source/data.txt", sep="   ", header=None, engine='python')
col_heading = ['Lever_position', 'Ship_speed', 'Gas_Turbine_shaft_torque', 'Gas_Turbine_rate_of_revolutions',
			   'Gas_Generator_rate_of_revolutions', 'Starboard_Propeller_Torque', 'Port_Propeller_Torque', 'HP_Turbine_exit_temperature',
			   'GT_Compressor_inlet_air_temperature', 'GT_Compressor_outlet_air_temperature', 'HP_Turbine_exit_pressure',
			   'GT_Compressor_inlet_air_pressure', 'GT_Compressor_outlet_air_pressure', 'Gas_Turbine_exhaust_gas_pressure',
			   'Turbine_Injecton_Control', 'Fuel_flow', 'GT_Compressor_decay_state_coefficient', 'GT_Turbine_decay_state_coefficient']
col_to_drop = ['Lever_position', 'Ship_speed', 'GT_Compressor_inlet_air_temperature', 'GT_Compressor_inlet_air_pressure']
final_cols = ['Gas_Turbine_shaft_torque', 'Gas_Turbine_rate_of_revolutions',
			  'Gas_Generator_rate_of_revolutions', 'Starboard_Propeller_Torque',
			  'Port_Propeller_Torque', 'HP_Turbine_exit_temperature',
			  'GT_Compressor_outlet_air_temperature', 'HP_Turbine_exit_pressure',
			  'GT_Compressor_outlet_air_pressure', 'Gas_Turbine_exhaust_gas_pressure',
			  'Turbine_Injecton_Control', 'Fuel_flow',
			  'GT_Compressor_decay_state_coefficient',
			  'GT_Turbine_decay_state_coefficient']

data.columns = list(col_heading)
data = data.drop(col_to_drop, axis=1)
X_train = data.drop(['GT_Compressor_decay_state_coefficient', 'GT_Turbine_decay_state_coefficient'], axis=1).values
y_train = data[col_heading[-2:]].values
X = data.drop(['GT_Compressor_decay_state_coefficient', 'GT_Turbine_decay_state_coefficient'], axis=1)
y1 = pd.DataFrame(data=y_train[:, 0], columns=[final_cols[-2]])
y2 = pd.DataFrame(data=y_train[:, 1], columns=[final_cols[-1]])
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
scaled_X = pd.DataFrame(data=X_train, columns=final_cols[:-2])
reg1 = NuSVR()
reg1.fit(X_train, y_train[:, 0])
reg2 = NuSVR()
reg2.fit(X_train, y_train[:, 1])
y_pred1 = reg1.predict(X=X_train)
y_pred2 = reg2.predict(X=X_train)
y_pred = np.hstack((y_pred1.reshape(-1, 1), y_pred2.reshape(-1, 1)))
printMetrics(y_true=y_train, y_pred=y_pred)
metrics = getMetrics(y_true=y_train, y_pred=y_pred)
fig1, ax1 = plt.subplots(figsize=(15, 15))
myplot1 = plot_partial_dependence(reg1, scaled_X, final_cols[:-2], ax=ax1, n_jobs=-1)
myplot1.plot()
fig1.savefig('GT_Compressor_decay_state_coefficient.png')
fig2, ax2 = plt.subplots(figsize=(15, 15))
myplot2 = plot_partial_dependence(reg2, scaled_X, final_cols[:-2], ax=ax2, n_jobs=-1)
myplot2.plot()
fig2.savefig('GT_Turbine_decay_state_coefficient.png')
