from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection


def read_split_data(iris_data):
    data = iris_data['data']
    targets = iris_data['target']
    train_data,test_data,train_targets,test_targets = model_selection.train_test_split(data,targets,test_size=0.1)
    return (train_data,test_data,train_targets,test_targets)


iris_dataset = datasets.load_iris()
train_data,test_data,train_targets,test_targets=read_split_data(iris_dataset)
