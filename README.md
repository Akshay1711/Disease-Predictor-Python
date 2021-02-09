
# Disease-Predictor-Python
## Prediction Process
The process of predicting the data in the test dataset consists of the following steps:
1. Extracted the csv using pandas library in python.
2. Ranged the values in both test and train dataset in between 20 and 16000, where 20 is the minimum value in the dataset and them maximum value is less than or equal to 16000. We have done this by modifying the values less than 20 to 20 and the values greater than 16000 to 16000. So that range of the values in the dataset will be in between 20 and 16000.
3. Acquired the minimum and maximum values of each row in the dataset and checked the fold difference for it. If we found the fold difference (i.e. max value in the row / min value in the row) less than 2 then we have removed those rows from the dataset, this step has been performed to remove the rows with identical values which will result the unique rows in the dataset which will be helpful for the better prediction of test data.
4. Divided the dataset from the previous step based on classes such that each class will have their corresponding columns and their row data and will be a subset of the main dataset.
5. Identified top 30 genes(rows) for each class subset by performing a 1 sample t-test on all the subsets and stored their indices in an array. As there are 5 classes, there are 5 arrays which consists of top 30 genes(rows) indices.
6. Once the top 30 genes(rows) indices are available from the previous step we have taken the top 2,4,6,8,10,12,15,20,25, and 30 indices values for each class and prepared the top N gene datasets which consists of top N gene indices data of all the classes without duplicate rows for which we have combined the top indices for each class and applied dict.fromkeys() which prepares a dictionary of indices values and removes duplicate indices, once duplicate indices are removed changed he dictionary of indices to list and prepared the top N gene datasets.
7. Once the top 2,4,6,8,10,12,15,20,25, and 30 gene subsets are prepared, we have transposed the data so that the samples are turned into rows and genes as columns and we appended a column of class values accordingly.
8. Once the top N subsets are prepared we have stored them as csv using pandas to_csv method with the naming convention mentioned in the requirements document.
9. Once top N datasets are prepared calculated accuracy and cross validation error for each top N dataset with Gaussian Naive Bayes, Decision tree, K Nearest Neighbour, Ada boost classifiers and Neural Network (MLP Classifier) and prepared a pandas dataframe consisting of the cross validation errors with cv folds as 5 and 2.
10. Checked for the lowest cross validation error in the whole dataframe and found that K Nearest Neighbour Classifier with n = 4 has provided the lowest cross validation error with top 25 genes dataset.
11. Once the top gene dataset and best classifier is acquired, prepared the test set with the list of indices with which the top 25 gene dataset is created.
12. Once the train and test set are finalized, we have predicted the test data and improved the accuracy by modifying the parameters of the K Nearest Neighbour classifier.
## Classifiers considered to process the dataset
### Gaussian Naive Bayes
Gaussian Naive Bayes classifier is one of the variants of Naive Bayes machine learning classification algorithm which follows Gaussian normal distribution with support for continuous data. It is a supervised machine learning algorithm which is based on Bayes theorem with a simple classification technique and is found useful with datasets having high dimensions. This classifier works by calculating a z-score difference between each point and each class mean (i.e. the distance of the class mean divided by the standard deviation of the corresponding class). Thus Gaussian Naive Bayes provides slightly different approach and can be used efficiently.

Library used: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

### Decision Tree Classifier
It is a supervised machine learning algorithm which follows a process known as binary recursive partitioning. Decision Tree classification is a iterative process of splitting the data into partitions based on a decision variable where variable is either categorical/ discrete, the process of splitting the data is done on each of the branches until they reach a pure leaf node on both sides of the tree where, pure leaf node means that samples at each leaf node will be of only a single class. 

Library used: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier

### K Nearest Neighbour Classifier
KNN is a type of supervised, lazy learning and non parametric learning algorithm which uses feature similarity to predict the values of new data points. The value of the new data point is assigned based on the best match with the points in the train set.

Library Used: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

### Ada-boost Classifier
Ada-boost or Adaptive Boosting Classifier is an ensemble boosting classifier which combines multiple classifiers to attain best accuracy on the datasets. This is an interactive ensemble method which build very strong classifiers by combining multiple poorly performing classifiers. The basic concept behind Ada-boost is that it sets the priority for the classifiers and iterates through them. In each iteration it ensures the accuracy of predictions of unusual observations, by which it attains highest accuracy and also minimizes the training error.

Library Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

### Neural Network (MLP Classifier)
Multilayer Perceptron Classifier (MLP Classifier) is a class of Feedforward Artificial Neural Network. MLP is a supervised algorithm which uses back propagation technique for processing the features of the dataset using various layers in it. There are atleast 3 layers which are essential to perform the classification, they are input layer, hidden layer (there can be one or more hidden layers which process the features) and an output layer.
Where,
Input Layer is the first layer which takes all the input features and the target.
Hidden Layers are the layers which processes the features.
Output Layer is the layer where the output is given by the classifier.
In a a neural network each feature which is being processed in the hidden layers are known as nodes (or) neurons which use a non-linear activation function to process their data.
The graph depicted below shows the cross validation error rate for Neural Network (MLP classifier):

Library Used: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

### Best Classifier and Geneset combination selected to predict test data
After calculating the accuracy and cross validation errors of generated Top N gene datasets across all the classifiers defined above, we have come to the conclusion that K Nearest Neighbor Classifier with n_neighbors as 3 provides a low error rate when compared with all the different classifiers we have used with the gene dataset consisting of 25 top genes data. Once we have found the best gene train dataset and corresponding classifier which has given a low cross validation error, we have prepared the test data based with the genes used in the best genes dataset, added more attributes to the classifier and predicted the test data.
