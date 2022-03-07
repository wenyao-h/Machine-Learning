# -*- coding: utf-8 -*-
"""
count the zero-one loss of decision tree with different values for parameter max_depth 
from [2,3,4,5,6,7,8,9,10], but random_state is always set to 0 
"""


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

# Load data
iris = load_iris()

# Determine feature matrix X and target array Y
X = iris.data
Y_true = iris.target

#++insert your code here++ to create and train different decision
# tree with different values for parameter max_depth 
# then count error and corresponding total number of nodes

def errorCounting(max_d):
    '''
    input:
    max_d: int, max_depth of the decision tree

    output:
    loss: int, number of errors
    loss + nodes: int, the SUM of number of nodes and number of errors
    '''
    print('='*15, f'max_depth: {max_d}','='*15)

    clf = DecisionTreeClassifier(max_depth=max_d, random_state=0)
    clf.fit(X, Y_true)
    Y_Pred = clf.predict(X)

    print("This trained decision tree has {} nodes".format(clf.tree_.node_count))

    count = Counter(Y_true==Y_Pred)
    loss = count.get(False, 0)
    print('Error: ', loss)
    return loss, loss+int(clf.tree_.node_count)



q1_list, q2_list = [], [] 
for i in range(2,11):
    q1, q2 = errorCounting(i)
    q1_list.append(q1)
    q2_list.append(q2)

print("The best depth when the decision tree generates least errors:", q1_list.index(min(q1_list))+2)
print("The best depth when the decision tree generates least sum of errors and nodes:", q2_list.index(min(q2_list))+2)

