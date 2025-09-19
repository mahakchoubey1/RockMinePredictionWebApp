import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


#1)load daataset
print ("Loading dataset")
dataset= pd.read_csv("copy of sonar data (1).csv")


print(dataset.columns)
# Features = all numeric columns (C1..C60)
X = dataset.drop(columns="C61", axis=1)

# Label = last column (Rock/Mine)
Y = dataset["C61"]
print("Shape of X:", X.shape)
print("Unique labels:", Y.unique())


# 2) Feature Selection (reduce from 60 → 10 best features)

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, Y)
selected_features = selector.get_support(indices=True)
print("🎯 Selected top features:", selected_features)

# 3) Train-test split

X_train,X_test,Y_train,Y_test=train_test_split(X_new,Y,test_size=0.1,random_state=2)

# 4) Model training

model=LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)

# 5) Evaluate model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# 6) Save model + selector (so UI can use same transformation)
with open("trained_model.pkl","wb")as f:
    pickle.dump((model, selector), f)
    print("💾 Model + feature selector saved as trained_model.pkl")




#%% md
#Importing the Dependencies
#%%
'''
#%%
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
#%% md
#Data Collection and Data Processing
#%%
#loading the dataset to a pandas Dataframe)
sonar_data = pd.read_csv('copy of sonar data(1).csv',header=None )
#%%
sonar_data.head()
#%%
sonar_data.tail()
#%%
# number of rows and columns
sonar_data.shape
#%%
sonar_data.describe()  #describe --> statistical measures of the data
#%%
sonar_data[60].value_counts()
#%%
sns.histplot( x=sonar_data[60].values, kde=True) # since no outliers exist so we can use mean for grouping data according to the label column
#%% md
#M --> Mine

#R --> Rock
#%%
sonar_data.groupby(60).mean()
#%%
# separating feature/data and target/Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
#%%
print(X)
print(Y)
#%% md
#Training and Test data
#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
#%%
print(X.shape, X_train.shape, X_test.shape)
#%%
print(X_train)
print(Y_train)
#%% md
#Model Training --> Logistic Regression
#%%
model = LogisticRegression()
#%%
#training the Logistic Regression model with training data
model.fit(X_train, Y_train)
#%% md
#Model Evaluation
#%%
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#%%
print('Accuracy on training data : ', training_data_accuracy)

#%%
cm_train=confusion_matrix(X_train_prediction, Y_train)
print(cm_train)


#%%
#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#%%
print('Accuracy on test data : ', test_data_accuracy)
#%%
cm_test=confusion_matrix(X_test_prediction,Y_test)
print(cm_test)
#%%


input_data = (1,0.0523,0.0653,0.0521,1.0611,0.0177,0.0665,1.0664,0.1460,0,0.3877,0.4992,0,0,0.5607,0,0,0.,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')


#saved the trained model

filename='trained_model.pkl'
#.dump function is used to sve model and wb is used to wrie and open file in binary format.

pickle.dump(model,open(filename,'wb'))

#loading the saved model
loader_model=pickle.load(open('trained_model.pkl'))'''
