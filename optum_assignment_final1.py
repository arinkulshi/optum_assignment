#!/usr/bin/env python
# coding: utf-8

# In[1]:



'''
Questions
1. Modify the training script so that only 80% of the data is used for training and the remaining 20% is test data.

2. Output the accuracy score of the model on the test data.

3. Implement a simple cross-validation step to find which of 1, 5, and 10 is the best max_depth for the classifier

4. Print the confusion matrix of the classifier that results from (3) using sklearn's built-in method. Which class has the most false positives?
'''






from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = load_iris()

features = data["data"]
labels = data["target"]


#confusion_matrix(labels, pred)

#Accuracy = (Number of elements correctly classified)/(Total elements)


# 
# 1. Modify the training script so that only 80% of the data is used for training and the remaining 20% is test data.
# 
# 2. Output the accuracy score of the model on the test data.
# 
# 3. Implement a simple cross-validation step to find which of 1, 5, and 10 is the best max_depth for the classifier
# 
# 4. Print the confusion matrix of the classifier that results from (3) using sklearn's built-in method. Which class has the most false positives?

# In[2]:


from sklearn.metrics import accuracy_score


#Modify the training script so that only 80% of the data is used for training and the remaining 20% is test data.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)

multi_class_value = (list(set(labels)))

no_of_total_observation = len(labels)

print (f'Total number of observations = {no_of_total_observation}, with multi-class label value {multi_class_value}')
print (f'Number of observations in Training set = {len(y_train)} and in Test set = {len(y_test)} after 80-20 split' )
#Implement a simple cross-validation step to find which of 1, 5, and 10 is the best max_depth for the classifier
#Output the accuracy score of the model on the test data.
max_depth_values = [1,5,10]
accuracies = {}
false_positives = []


# In[3]:


"""During my repeated training and validation process (which is in the for loop below) on the same train and test split
I get a small variation on test prdiction and test accuracy. Let me attach that for your refererence. 
This dictionary has two elements; Accuracy Score value and the confusion matrix.

{1: [0.6333333333333333, array([
	   [12,  0,  0],
       [ 0,  0, 11],
       [ 0,  0,  7]], dtype=int64)], 
 5: [0.9666666666666667, array([
       [12,  0,  0],
       [ 0, 11,  0],
       [ 0,  1,  6]], dtype=int64)], 
 10: [1.0, array([
       [12,  0,  0],
       [ 0, 11,  0],
       [ 0,  0,  7]], dtype=int64)]}
	   
{1: [0.6333333333333333, array([
	   [12,  0,  0],
       [ 0,  0, 11],
       [ 0,  0,  7]], dtype=int64)], 
 5: [0.9666666666666667, array([
       [12,  0,  0],
       [ 0, 11,  0],
       [ 0,  1,  6]], dtype=int64)], 
 10: [0.9666666666666667, array([
       [12,  0,  0],
       [ 0, 11,  0],
       [ 0,  1,  6]], dtype=int64)]}

	   
{1: [0.6333333333333333, array([
	   [12,  0,  0],
       [ 0,  0, 11],
       [ 0,  0,  7]], dtype=int64)], 
 5: [1.0, array([
	   [12,  0,  0],
       [ 0, 11,  0],
       [ 0,  0,  7]], dtype=int64)], 
 10: [0.9666666666666667, array([
	   [12,  0,  0],
       [ 0, 11,  0],
       [ 0,  1,  6]], dtype=int64)]}
"""



for i in max_depth_values:
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(X_train, y_train)
    prediction_test_data = model.predict(X_test)
    #accuracies.append(accuracy_score(y_test,prediction_test_data))
    accuracies_list = []
    #accuracies_list.append(accuracy_score(y_test,prediction_test_data))
    #accuracies_list.append(confusion_matrix(y_test,prediction_test_data))
    conf_matrix = confusion_matrix(y_test,prediction_test_data)
    accuracies[i] = [accuracy_score(y_test,prediction_test_data),confusion_matrix(y_test,prediction_test_data)]
#print(accuracies)    


# In[7]:



'''
Please note that the confusion matrix in scikit-learn has been generated using 
Column_value = Predicted value and Row_value = Actual value. 
let me describe the computation logic for False Positive values for the the classes. 
Please note that we have 3 classes here for predicted Y values; 0,1,2
Example for calculation for false positive for the class 0,1 and 2
false_positive_class0 =confusion_matrix[1][0] + confusion_matrix[2][0]
false_positive_class1 =confusion_matrix[0][1] + confusion_matrix[2][1]
false_positive_class2 =confusion_matrix[0][2] + confusion_matrix[1][2]

Please note that this is a column based aggegation
'''


for inxi,accuracy in accuracies.items():
#Initializing the counter that holds false positive for three predicted class values i.e. 0,1,2    
    false_positive = [0] * len(multi_class_value)

#Getting the confusion matrix element from the accuracies dictionary which is the 2nd element    
    confusion_matrix = accuracy[1]
    for inxj, actual_value_row  in enumerate(confusion_matrix):
        for inxk, predicted_value in enumerate(actual_value_row):
            if inxj == inxk:
                pass
            else:
                false_positive[inxk] = false_positive[inxk] + predicted_value
    print(f'Following are the results from Decision Tree model with Max Depth Value {inxi}')
    print(f'=======================================================================')
    print(f'Accuracy score : {accuracy[0]}')
    print(f'Confusion matrix for the max depth value {inxi} of Decision Tree model')
    print(f'---------------------------------------------------------------------')
    print(f'Prediction count for different classes, Column represents Predicted value, Row represents Actual Value')
    print(f'  Y={multi_class_value[0]} Y={multi_class_value[1]} Y={multi_class_value[2]}')
    print('\n'.join([''.join(['{:4}'.format(true_false_count) for true_false_count in row_counts]) for row_counts in accuracy[1]]))
    
    print(f'False positive prediction from Decision Tree model {false_positive} for corresponding Y values: {multi_class_value}' )
    max_false_positive_value = max(false_positive)
    print(f'Maximum False positive found for class {false_positive.index(max_false_positive_value)} with False-Positive count {max_false_positive_value} \n\n')


#for values in accuracies.items():
    #print(values[1][0][1])

#Print the confusion matrix of the classifier that results from (3) using sklearn's built-in method. Which class has the most false positives?

#{1:}

