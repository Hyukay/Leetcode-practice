import numpy as np
from collections import Counter
def euclideanDistance(x1,x2):

    return np.sqrt(np.sum((x1-x2)**2))






class KNN:
    
    def __init__(self, k):
        self.k = k
    

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y


    def predict(self,X):

        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):

        #calculate the distances between the point x and all points in the train set
        distances = [euclideanDistance(x,x_train) for x_train in X_train]
        #get k nearest samples (so this goes until k)
        #and k = 3 in our example here but could be anything else
        k_indices = np.argsort(distances)[0:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #majority vote, most common class label
        #use counter most.common()
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]
    


        
