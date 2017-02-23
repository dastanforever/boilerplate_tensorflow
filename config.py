#######################################
#
#	Boilerplate for tensorflow
#	Neural Network Model.
#
#######################################

from utils import exceptions

class Config:
	"""This class is used to store
		hyperparameters. 
	"""
	def __init__(self, n_features, n_classes, dropout=0.5,\
					batch_size=2048, n_epochs=1000, learning_rate=0.001):
		self.config_dict = {
			self.n_features = n_features,
			self.n_classes = n_classes,
			self.dropout = dropout,
			self.batch_size = batch_size,
			self.n_epochs = n_epochs,
			self.learning_rate = learning_rate
		}

	def add_config(self, param_name, param_value):
		# adds value, raises if already present.
		if param_name in self.config_dict:
			raise exceptions.KeyExistsInDict
		self.config_dict[param_name] = param_value

	def update_config(self, param_name, param_value):
		self.config_dict[param_name] = param_value