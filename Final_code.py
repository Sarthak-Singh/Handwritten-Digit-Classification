import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("train.csv").to_numpy()

clf1 = KNeighborsClassifier()           # K Nearest Neighbor Classifier
clf2 = DecisionTreeClassifier()         # Decision Tree Classifier
clf3 = SVC()                            # Support Vector Classifier
clf4 = GaussianNB()                     # Naive Baye's Classifier

# Training Dataset
xtrain = data[0:36000, 1:]
train_label = data[0:36000, 0]

# Training the classification models
clf1.fit(xtrain, train_label)
clf2.fit(xtrain, train_label)
clf3.fit(xtrain, train_label)
clf4.fit(xtrain, train_label)

# Testing Data
xtest = data[36000:, 1:]
actual_label = data[36000:, 0]

# Measurement of Accuracy of Classification Models
p1 = clf1.predict(xtest)
p2 = clf2.predict(xtest)
p3 = clf3.predict(xtest)
p4 = clf4.predict(xtest)

count1 = count2 = count3 = count4 = 0
for i in range(0, 6000):
    count1 += 1 if p1[i] == actual_label[i] else 0
    count2 += 1 if p2[i] == actual_label[i] else 0
    count3 += 1 if p3[i] == actual_label[i] else 0
    count4 += 1 if p4[i] == actual_label[i] else 0

a1 = (count1 / 6000) * 100
a2 = (count2 / 6000) * 100
a3 = (count3 / 6000) * 100
a4 = (count4 / 6000) * 100

pt.bar(["K Nearest Neighbor Classifier", "Decision Tree Classifier", "Support Vector Classifier",
        "Naive Baye's Classifier"], [a1, a2, a3, a4], color=["red", "green", "blue", "yellow"])
pt.xlabel("Type of Classifier")
pt.ylabel("Accuracy (in %)")
pt.title("Comparison in Accuracy")
pt.show()
