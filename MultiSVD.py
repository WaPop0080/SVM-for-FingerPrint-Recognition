import sklearn.model_selection
from sklearn.datasets import make_blobs
from sklearn import svm
import joblib
import numpy as np
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataLoader import readData
from sklearn.decomposition import PCA
from dataLoader import writeData
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt


path_train = r'./DB3_feature/features2.txt'
path_test = r'./DB3_feature/features2_test.txt'

# train_features = [[1, 1, 1], [0, 1, 0], [3, 5, 3]]
# train_label = [0, 1, 2]
# test_features = [[1.5, 2, 1], [2.5, 3, 2], [4, 7, 3]]
# test_label = [0, 1, 2]

train_features = []
train_label = []
test_features = []
test_label = []

train_set = readData(path_train)
test_set = readData(path_test)

for i in range(len(train_set)):
    train_feature = []
    for j in range(len(train_set[i]) - 1):
        train_feature.append(train_set[i][j])
    train_features.append(train_feature)
    train_label.append(train_set[i][-1])

for i in range(len(test_set)):
    test_feature = []
    for j in range(len(test_set[i]) - 1):
        test_feature.append(test_set[i][j])
    test_features.append(test_feature)
    test_label.append(test_set[i][-1])

train_features = numpy.array(train_features)
train_label = numpy.array(train_label)
test_features = numpy.array(test_features)
test_label = numpy.array(test_label)

# print(len(train_features))
# print(len(train_label))
# print(len(train_features[0]))
# print(len(test_features))
# print(len(test_label))
# print(len(test_features[0]))

# pca = PCA(n_components=0.80)
# pca.fit(train_features)
# print(pca.n_components_)
# train_features = pca.transform(train_features)
# test_features = pca.transform(test_features)


clf = svm.SVC(C=0.8, gamma=0.01, kernel='linear', max_iter=40000, decision_function_shape='ovo')

clf.fit(train_features, train_label)

# Test on Training data
train_result = clf.predict(train_features)
precision = sum(train_result == train_label) / train_label.shape[0]
print('Training precision: ', precision)

# Test on test data
test_result = clf.predict(test_features)

print(test_result)
print(test_label)
precision = sum(test_result == test_label) / test_label.shape[0]
print('Test precision: ', precision)

svm_pip = Pipeline([('scaler', StandardScaler()), ('svm', svm.SVC(C=0.8, gamma=0.2, kernel='sigmoid', max_iter=40000))])
scores = cross_val_score(svm_pip, train_features, train_label, cv=3, scoring='accuracy')
print("scores:{}, scores mean:{} +/- {}".format(scores, np.mean(scores), np.std(scores)))
train_sizes, train_scores, test_scores = learning_curve(svm_pip, train_features, train_label, cv=3,
                                                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 14),
                                                        scoring="accuracy")
train_scores_mean = np.mean(train_scores,axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="training")
plt.plot(train_sizes,test_scores_mean, 'o-', color="b", label="Cross-validation")
plt.xlabel("training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()

# joblib.dump(clf, './clf.pkl')
