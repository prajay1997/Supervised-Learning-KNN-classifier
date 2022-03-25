import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\K nearest neighbour(KNN) classifier\datasets\glass.csv")
data.shape
data.columns
data.describe()

data.isna().sum()

# normalized the data
def norm_fun(i):
     x = (i-i.min())/(i.max()-i.min())
     return(x)

data_norm = norm_fun(data.iloc[:,0:9])


# univariate visualisation
sns.distplot(data['Type'], kde=False)

X = np.array(data_norm.iloc[:,:])    # predictors
Y = np.array(data['Type'])      # target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

# Imbalance check 
 data.Type.value_counts()

# converting data from array to dataframe 

Y_train = pd.DataFrame(Y_train)
Y_test  = pd.DataFrame(Y_test)

Y_train.value_counts()import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\K nearest neighbour(KNN) classifier\datasets\glass.csv")
data.shape
data.columns
data.describe()

data.isna().sum()

# normalized the data
def norm_fun(i):
     x = (i-i.min())/(i.max()-i.min())
     return(x)

data_norm = norm_fun(data.iloc[:,0:9])


# univariate visualisation
sns.distplot(data['Type'], kde=False)

X = np.array(data_norm.iloc[:,:])    # predictors
Y = np.array(data['Type'])      # target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

# Imbalance check 
 data.Type.value_counts()

# converting data from array to dataframe 

Y_train = pd.DataFrame(Y_train)
Y_test  = pd.DataFrame(Y_test)

Y_train.value_counts()
Y_test.value_counts()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)

knn.fit(X_train , Y_train)

#  Evaluate the model
pred = knn.predict(X_test)
pred

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Y_test, pred))
confusion_matrix( Y_test, pred)
result = classification_report(Y_test, pred)
 
 # error on train data
 pred_train = knn.predict(X_train)
 print(accuracy_score(Y_train, pred_train))
confusion_matrix(Y_train, pred_train)
result1 = classification_report(Y_train, pred_train)
print(result1)
# creating empty list variable
acc =[]

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc  = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])
   
    
   # train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
import matplotlib.pyplot as plt # library to do visualizations 

# from the plot we K= 3 gives the good accuracy to the model

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train , Y_train)

#  Evaluate the model
pred = knn.predict(X_test)
pred

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Y_test, pred))
confusion_matrix( Y_test, pred)
result = classification_report(Y_test, pred)
 
 # error on train data
 pred_train = knn.predict(X_train)
 print(accuracy_score(Y_train, pred_train))
confusion_matrix(Y_train, pred_train)
result1 = classification_report(Y_train, pred_train)
print(result1)



Y_test.value_counts()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)

knn.fit(X_train , Y_train)

#  Evaluate the model
pred = knn.predict(X_test)
pred

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Y_test, pred))
confusion_matrix( Y_test, pred)
result = classification_report(Y_test, pred)
 
 # error on train data
 pred_train = knn.predict(X_train)
 print(accuracy_score(Y_train, pred_train))
confusion_matrix(Y_train, pred_train)
result1 = classification_report(Y_train, pred_train)
print(result1)
# creating empty list variable
acc =[]

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc  = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])
   
    
   # train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
import matplotlib.pyplot as plt # library to do visualizations 

# from the plot we K= 3 gives the good accuracy to the model

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train , Y_train)

#  Evaluate the model
pred = knn.predict(X_test)
pred

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Y_test, pred))
confusion_matrix( Y_test, pred)
result = classification_report(Y_test, pred)
 
 # error on train data
 pred_train = knn.predict(X_train)
 print(accuracy_score(Y_train, pred_train))
confusion_matrix(Y_train, pred_train)
result1 = classification_report(Y_train, pred_train)
print(result1)


