import numpy as np
import matplotlib as plt
from matplotlib import ListedColormap

iris = datasets.load.iris()


X, y = iris.data, iris.target


X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4487)



print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train)

plt.figure()

plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolor='k', s=20)

plt.show()


from knn import KNN


#create classifier

clf = knn(k=3)

clf.fit(X_train, y_train)


#want to calculate with our test set
predictions = clf.predict(X_test)


accuracy = np.sum(predictions==y_test)/ len(y_test)

print(accuracy) 