import graphviz as graphviz
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

np.random.seed(5)

'''
    purchase predictor
    from pages accessed by users, predict if they are going to buy something
'''
# # get data #
# df = pd.read_csv('tracking.csv')
# x = df[["home", "how_it_works", "contact"]]
# y = df["bought"]
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y)
#
# # predict #
# model = LinearSVC()
# model.fit(train_x, train_y)
# predict = model.predict(test_x)
#
# score = accuracy_score(test_y, predict)
#
# # output #
# print("Accuracy of the model is: %.2f%%" % (score * 100))

'''
    project predictor
    from time and price for each project, predict if its going to be taken
'''
# get data #
# df = pd.read_csv('projects.csv')
# m = {
#     0: 1,
#     1: 0
# }
#
# df['finished'] = df.unfinished.map(m)
# df = df.drop(columns=['unfinished'])
#
# x = df[["expected_hours", "price"]]
# y = df["finished"]
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y)
# baseline = np.ones(len(test_y))

# x_min, x_max = test_x.expected_hours.min(), test_x.expected_hours.max()
# y_min, y_max = test_x.price.min(), test_x.price.max()
# points = 100
# x_axis = np.arange(x_min, x_max, (x_max - x_min) / points)
# y_axis = np.arange(y_min, y_max, (y_max - y_min) / points)
# xx, yy = np.meshgrid(x_axis, y_axis)
# coord = np.c_[xx.ravel(), yy.ravel()]
#
# # predict #
# model = LinearSVC()
# model.fit(train_x, train_y)
# predict = model.predict(test_x)
# z = model.predict(coord)
# z = z.reshape(xx.shape)
#
# score = accuracy_score(test_y, predict)
# baseline_score = accuracy_score(baseline, predict)
#
# # plot #
# sns.relplot(x='expected_hours', y='price', hue='finished', data=df)
# sns.relplot(x='expected_hours', y='price', hue=test_y, data=test_x)
# sns.relplot(x='expected_hours', y='price', hue=predict, data=test_x)
# plt.contourf(x_axis, y_axis, z, alpha=0.5)
#
# # output #
# print(predict.sum())
# print("Stratify: %d 1's in train and %d 1's in test" % (train_y.sum(), test_y.sum()))
# print("Training with %d elements and testing with %d" % (len(train_y), len(test_y)))
# print("Accuracy of the model is: %.2f%%" % (score * 100))
# print("Accuracy of the baseline is: %.2f%%" % (baseline_score * 100))
# plt.show()

'''
    nonlinear classifier for dataset above
'''
# # get data #
# df = pd.read_csv('projects.csv')
# m = {
#     0: 1,
#     1: 0
# }
#
# df['finished'] = df.unfinished.map(m)
# df = df.drop(columns=['unfinished'])
#
# x = df[["expected_hours", "price"]]
# y = df["finished"]
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y)
# baseline = np.ones(len(test_y))
#
# x_min, x_max = test_x.expected_hours.min(), test_x.expected_hours.max()
# y_min, y_max = test_x.price.min(), test_x.price.max()
# points = 100
# x_axis_original = np.linspace(x_min, x_max, points)
# y_axis_original = np.linspace(y_min, y_max, points)
#
# # scale #
# scaler = StandardScaler()
# scaler.fit(train_x)
# scaled_train_x = scaler.transform(train_x)
# scaler.fit(test_x)
# scaled_test_x = scaler.transform(test_x)
#
# x_min, x_max = scaled_test_x[:,0].min(), scaled_test_x[:,0].max()
# y_min, y_max = scaled_test_x[:,1].min(), scaled_test_x[:,1].max()
# x_axis = np.linspace(x_min, x_max, points)
# y_axis = np.linspace(y_min, y_max, points)
# xx, yy = np.meshgrid(x_axis, y_axis)
# coord = np.c_[xx.ravel(), yy.ravel()]
#
# # predict #
# model = SVC()
# model.fit(scaled_train_x, train_y)
# predict = model.predict(scaled_test_x)
# z = model.predict(coord)
# z = z.reshape(xx.shape)
#
# score = accuracy_score(test_y, predict)
# baseline_score = accuracy_score(baseline, predict)
#
# # plot #
# sns.relplot(x='expected_hours', y='price', hue=predict, data=test_x, s=16)
# plt.contourf(x_axis_original, y_axis_original, z, alpha=0.5)
#
# # output #
# print("Stratify: %d 1's in train and %d 1's in test" % (train_y.sum(), test_y.sum()))
# print("Training with %d elements and testing with %d" % (len(train_y), len(test_y)))
# print("Accuracy of the model is: %.2f%%" % (score * 100))
# print("Accuracy of the baseline is: %.2f%%" % (baseline_score * 100))
# plt.show()

'''
    car predictor
    predict if a car is going to be sold or not (based on year, price and mileage)
'''
# # get data #
# KM_PER_MILE = 1.60934
# df = pd.read_csv('car-prices.csv')
# m = {
#     'yes': 1,
#     'no': 0
# }
# df['sold'] = df.sold.map(m)
# df['model_age'] = datetime.datetime.today().year - df.model_year
# df['km_per_year'] = df.mileage_per_year * KM_PER_MILE
#
# x = df[["km_per_year", "model_age", "price"]]
# y = df["sold"]
#
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y)
#
# # scale #
# scaler = StandardScaler()
# scaler.fit(train_x)
# scaled_train_x = scaler.transform(train_x)
# scaler.fit(test_x)
# scaled_test_x = scaler.transform(test_x)
#
# # predict #
# dummy = DummyClassifier(strategy="stratified")
# dummy.fit(scaled_train_x, train_y)
# baseline_score = dummy.score(scaled_test_x, test_y)
#
# model = SVC()
# model.fit(scaled_train_x, train_y)
# score = model.score(scaled_test_x, test_y)
#
#
# # output #
# print("Training with %d elements and testing with %d" % (len(train_y), len(test_y)))
# print("Accuracy of the model is: %.2f%%" % (score * 100))
# print("Accuracy of the baseline is: %.2f%%" % (baseline_score * 100))

'''
    same problem above with a decision tree classifier
'''
# # get data #
# KM_PER_MILE = 1.60934
# df = pd.read_csv('car-prices.csv')
# m = {
#     'yes': 1,
#     'no': 0
# }
# df['sold'] = df.sold.map(m)
# df['model_age'] = datetime.datetime.today().year - df.model_year
# df['km_per_year'] = df.mileage_per_year * KM_PER_MILE
#
# x = df[["km_per_year", "model_age", "price"]]
# y = df["sold"]
#
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y)
#
# # predict #
# model = DecisionTreeClassifier(max_depth=5)
# model.fit(train_x, train_y)
# score = model.score(test_x, test_y)
#
# # output #
# print("Training with %d elements and testing with %d" % (len(train_y), len(test_y)))
# print("Accuracy of the model is: %.2f%%" % (score * 100))
# dot_data = export_graphviz(model, out_file=None, filled=True, feature_names=x.columns, rounded=True, class_names=["no", "yes"])
# g = graphviz.Source(dot_data)
# g.render(view=True)

