# Artificial Neural Network

""" The business is to predict whether the customer of a bank is going to quit or not through some independent 
features like 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
'IsActiveMember', 'EstimatedSalary' by ANN so that that could help the bank to rank the customer based on
the probability of person to churn.""" 

##importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
## taking the independent and dependent variables for our model
x = pd.DataFrame(dataset.iloc[:,3:13].values)
y = dataset.iloc[:,-1].values

##dealing with categorical variables(encoding categorical variales)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
x.iloc[:,1] = labelencoder_x1.fit_transform(x.iloc[:,1])
labelencoder_x2 = LabelEncoder()
x.iloc[:,2] = labelencoder_x2.fit_transform(x.iloc[:,2])
onehotencoder_x1 = OneHotEncoder(categorical_features = [1])
x = onehotencoder_x1.fit_transform(x).toarray()
x = x[:,1:]

##splitting the data in to training, testing 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

##feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

##importing keras libraray and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

##initializing ANN model
classifier = Sequential()
# We are initializing the classifier

##adding input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
## relu is the 'rectifier' activation function used.
## there is no specific rule for number of nodes in hidden layer, its just an art. The tip would be taking the 
##average of input and output layers average (ex: 11 indepenedent variables -> 11 input nodes. and 1 output node)
## average = (11 + 1)/2 = 6
## init = 'uniform' states that the initial assigned weights are uniform.

##adding another hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))
## in second hidden layer we don't need t give the inpit_dim because the output of hidden layer1 is taken as default

## adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
## since it is classification of twwo we use output_dim = 1, and activation function = sigmoid
##if classification is for 5 classes we use output_dim =5

##compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
## here adam specifies that the model used for optimization is stochastic gradient descent batch model.
## loss = 'binary_crossentropy' states the loss function used sinse it is logstic regression the specific is used.
##accuracy is the term up on which the model is evaluted for each apoc

##fitting the model to the training data
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

##predictions using our ANN model
y_pred = classifier.predict(x_test)
## here we get y_pred the probability so, we convert it in to wheather the custemer leave or not by giving threshold
y_pred = (y_pred > 0.5)

##creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)







