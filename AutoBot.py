from AdaBoostRegressor import AdaBoost, AdaBoostGS
from DecisionTreeRegressor import DecisionTree, DecisionTreeGS
from ElasticNet import ElasticNetRegressor, ElasticNetRegressorGS
from ExtraTreeRegressor import ExtraTree, ExtraTreeGS
from GradientBoostingRegressor import GradientBoosting, GradientBoostingGS
from Lars import LarsRegressor, LarsRegressorGS
from Lasso import LassoRegressor, LassoRegressorGS
from LinearSVR import LinearSVRRegressor, LinearSVRRegressorGS
from MLPRegressor import NeuralNet, NeuralNetGS
from NuSVR import NuSVRRegressor, NuSVRRegressorGS
from PoissonRegressor import PoissonReg, PoissonRegGS
from Ridge import RidgeRegressor, RidgeRegressorGS
from SGDRegressor import SGD, SGD_GS
from SVR import SVRRegressor, SVRRegressorGS
from Utility import getPlantsPropulsionData


def Normal(makePolynomialFeatures):
	for i in range(5):
		print("*" * 20, i)
		X_train, X_test, y_train, y_test = getPlantsPropulsionData(splitData=True, makePolynomialFeatures=makePolynomialFeatures)
		AdaBoost(X_train, X_test, y_train, y_test)
		DecisionTree(X_train, X_test, y_train, y_test)
		ElasticNetRegressor(X_train, X_test, y_train, y_test)
		ExtraTree(X_train, X_test, y_train, y_test)
		GradientBoosting(X_train, X_test, y_train, y_test)
		LarsRegressor(X_train, X_test, y_train, y_test)
		LassoRegressor(X_train, X_test, y_train, y_test)
		LinearSVRRegressor(X_train, X_test, y_train, y_test)
		NeuralNet(X_train, X_test, y_train, y_test)
		NuSVRRegressor(X_train, X_test, y_train, y_test)
		PoissonReg(X_train, X_test, y_train, y_test)
		RidgeRegressor(X_train, X_test, y_train, y_test)
		SGD(X_train, X_test, y_train, y_test)
		SVRRegressor(X_train, X_test, y_train, y_test)


def GridSearch(makePolynomialFeatures):
	for i in range(1):
		print("*" * 20, i)
		X_train, X_test, y_train, y_test = getPlantsPropulsionData(splitData=True, makePolynomialFeatures=makePolynomialFeatures)
		AdaBoostGS(X_train, X_test, y_train, y_test)
		DecisionTreeGS(X_train, X_test, y_train, y_test)
		ElasticNetRegressorGS(X_train, X_test, y_train, y_test)
		ExtraTreeGS(X_train, X_test, y_train, y_test)
		GradientBoostingGS(X_train, X_test, y_train, y_test)
		LarsRegressorGS(X_train, X_test, y_train, y_test)
		LassoRegressorGS(X_train, X_test, y_train, y_test)
		LinearSVRRegressorGS(X_train, X_test, y_train, y_test)
		NeuralNetGS(X_train, X_test, y_train, y_test)
		NuSVRRegressorGS(X_train, X_test, y_train, y_test)
		PoissonRegGS(X_train, X_test, y_train, y_test)
		RidgeRegressorGS(X_train, X_test, y_train, y_test)
		SGD_GS(X_train, X_test, y_train, y_test)
		SVRRegressorGS(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
	# Normal(True)
	GridSearch(True)
# Normal(False)
