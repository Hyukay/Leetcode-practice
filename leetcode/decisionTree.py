

def entropy(labels):
    #ex labels = [0,0,0,0,0,1,1,1,1]
    import numpy as np
    from collections import Counter

    
    labels_count = Counter(labels)
    

    total_count = len(labels)

    entropy = -np.sum([(p/total_count)*np.log2(p/total_count) for p in labels_count if p>0]) 

    return entropy

class Node:

    def __init__(self,feature= None, treshold = None, left= None, right = None,*,value=None):

        self.feature = feature
        self.treshold = treshold
        self.left = left
        self.right = right
        self.value = value


    def is_leaf_node(self):

        if self.value is not None:
            return True
        
        return False

class DecisionTree:

    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    #X is training data 
    #y is training labels
    def fit(self,X,y):

        #apply safety check 1 because the 2nd dimension is the nb of features and makes sure that it can never be greater than the nb of features
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        #grow tree
        self.root = self._grow_tree(X,y)
        
    def _grow_tree(self, X, y, depth = 0):
        import numpy as np
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        #stopping criteria
        if (depth >= self.max_depth 
            or n_labels == 1 
            or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value = leaf_value)
        
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        #greedy search 
        best_feat, best_tresh = self._best_criteria(X,y,feature_idxs)

        left_idxs, right_idxs = self._split(X[:,best_feat], best_tresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idx], depth+1)
        return Node(best_feat,best_tresh,left,right)
    
    def predict(self, X):
        import numpy as np
        return np.array([self._traverse_tree(x) for x in X])
    

    def _traverse_tree(self,x):
        

    def _best_criteria(self,X,y, feature_idxs):

        import numpy as np

        best_gain = -1

        split_idx, split_tresh = None, None

        for feature_idx in feature_idxs:

            X_Column = X[:, feature_idx]
            tresholds = np.unique(X_Column)
            for treshold in tresholds:
                gain = self._information_gain(y,X_Column,treshold)
                if gain> best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_tresh = treshold

        
        return split_idx, split_tresh
    


    def _information_gain(self, y, X_Column, split_treshold):

        #parent entropy
        parent_entropy = entropy(y)

        #generate a split
        l_idxs, r_idxs = self._split(X_Column, split_treshold)

        if len(l_idxs) == 0 or len(r_idxs) == 0:
            return 0
        #calculate weighted average of the child entropy
        n = len(y)
        nb_left_samples, nb_right_samples = len(l_idxs), len(r_idxs)
        entropy_left, entropy_right = entropy(y[l_idxs]), entropy(y[r_idxs])

        child_entropy = (nb_left_samples/n) * entropy_left + (nb_right_samples/n) * entropy_right


        information_gain = parent_entropy - child_entropy
        return information_gain



    def _split(self, X_Column, split_treshold):
        import numpy as np
        left_idxs = np.argwhere(X_Column <= split_treshold).flatten()
        right_idxs = np.argwhere(X_Column > split_treshold).flatten()

        return left_idxs, right_idxs


    def predict(self,X):

        #traverse tree

    def most_common_label(labels):

        from collections import Counter
        most_common_labels = Counter(labels)

        most_common_label = most_common_labels.most_common(1)[0][0]

        return most_common_label





