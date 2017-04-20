import numpy as np
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

class Datasets:
	def __init__(self, X_train, y_train, X_test=None, y_test=None, test_split=0.4, train_dev_split=None, dev_split=0.5, small=None,
					X_transforms={}, y_transforms={}, stratify=None):

		# If stratify is given, check that it's a tuple of classes for train and test
		stratify_train, stratify_test = None, None
		if stratify:
			if not isinstance(stratify, tuple) or len(stratify)!=2:
				raise Exception("stratify should be a tuple of 2 numpy arrays: (training_classes, test_classes)")
			if stratify[0].shape[0] != y_train.shape[0]:
				raise Exception("stratify[0] should have the same number of elements as y_train")
			if stratify[1].shape[0] != y_test.shape[0]:
				raise Exception("stratify[1] should have the same number of elements as y_test")

			stratify_train = stratify[0]
			stratify_test = stratify[1]

		# If no testset - split training.
		if (X_test is None and y_test is None):
			X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_split, stratify=stratify_train)

		# if small was given - also create a 'small' version of the whole datasets under 'small'
		if small:
			_, X_train_small, _, y_train_small = train_test_split(X_train, y_train, test_size=small, stratify=stratify_train)
			_, X_test_small, _, y_test_small = train_test_split(X_test, y_test, test_size=small, stratify=stratify_test)

		# If train_dev split was given - split train into 'train' and 'train_dev'
		if train_dev_split:
			X_train, X_train_dev, y_train, y_train_dev = train_test_split(X_train, y_train, test_size=train_dev_split, stratify=stratify_train)

		# If dev_split was given - split test into 'dev' and 'test'
		if dev_split:
			X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=dev_split, stratify=stratify_test)

		# set all datasets
		self.train = Dataset(X_train, y_train, X_transforms, y_transforms)
		self.test = Dataset(X_test, y_test, X_transforms, y_transforms)
		if train_dev_split: self.train_dev = Dataset(X_train_dev, y_train_dev, X_transforms, y_transforms)
		if dev_split:       self.dev = Dataset(X_dev, y_dev, X_transforms, y_transforms)
		if small:           self.small = Datasets(X_train_small, y_train_small, X_test_small, y_test_small,
												 train_dev_split=None, dev_split=None, small=None,
												 X_transforms=X_transforms, y_transforms=y_transforms)

	def show(self):
		ordered_datasets = ['train', 'train_dev', 'dev', 'test', 'small']
		for ds_name in ordered_datasets:
			if not ds_name in self.__dict__: continue

			print(ds_name)
			self.__dict__.get(ds_name).show()


	def evaluate(self, model, X_transform_name=None, y_transform_name=None, post_predict_func=None):
		ordered_datasets = ['train', 'train_dev', 'dev', 'test']

		print("{:15} {:15}".format("Dataset", "Accuracy"))
		print("-"*15 + " " + "-"*15)
		for ds_name in ordered_datasets:
			if not ds_name in self.__dict__: continue
			ds = self.__dict__.get(ds_name)

			X = ds.__dict__["X_{}".format(X_transform_name)] if X_transform_name else ds.X
			y = ds.__dict__["y_{}".format(y_transform_name)] if y_transform_name else ds.y

			predicts = model.predict(X)
			if post_predict_func:
				predicts = post_predict_func(predicts)
			score = accuracy_score(predicts, y)

			print("{:15} {:15.4f}".format(ds_name, score))





class Dataset:
	def __init__(self, X, y, X_transforms, y_transforms):
		self.X = X
		self.y = y

		for tname, tfunc in X_transforms.items():
			self.__dict__["X_{}".format(tname)] = tfunc(X)
		for tname, tfunc in y_transforms.items():
			self.__dict__["y_{}".format(tname)] = tfunc(y)

	def show(self):
		for x in sorted(self.__dict__.keys()):
			print("  - {}: {}".format(x, self.__dict__.get(x).shape))
