---
title: "ARTIFICIAL INTELLIGENCE FOR HUMAN RESOURCES :  Model to reduce hiring and training costs of employees"
author: "Dr LEBEDE  Ngartera"
Update: "1--05--2024"
output:
  html_document:
    toc: yes
    number_sections: yes
  pdf_document:
    toc: yes
---
<p align="center">
  <img src="Screenshot 2024-01-10 182119.png"></b><br>
</p>

# ARTIFICIAL INTELLIGENCE FOR HUMAN RESOURCES :  **"Model to reduce hiring and training costs of employees"**

## I-OVERVIEW

This project develops an AI model to reduce hiring and training costs of employees by predicting which employees might leave the company. It is designed to manage data in a multinational company. The human resources team has collected a lot of data on its employees. We are going to develop a model that can predict which employees are most likely to quit. We are going to harness Python's power with this libraries: Pandas, Numpy, Matplotlib, Seborn to maniplating, handling missing data, cleaning, do numerical operations and making statistical graphics. KDE (Kernel Density Estimate) is used for visualizing the Probability Density of a continuous variable. Logistic Regression classifier (sklearn.linear_model), Random Classifier (confusion_matrix) and Deep Learning Model (Tensorflow, Keras) are used for training and evaluating.

We are dealing in this project with a large amount of data. Here is a sample of the following data set:

1. Education
2. JobInvolvement
3. JobSatisfaction
4. PerformationRating
5. WorkLifeBalance
6. EnvironmentSatisfaction
7. BusinessTravel ...
   
## II-IMPORT LIBRARIES AND DATASETS
Data source https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Thanks a million to the Kaggle Team for providing this data.

<p align="center">
  <img src="Screenshot 2024-01-10 182045.png">
</p>
<p align="center">
  <img src="Screenshot 2024-01-10 181957.png">
</p>


employee_df.head(7)
<p align="center">
  <img src="Screenshot 2024-01-10 181937.png">
</p>

employee_df.tail(10)

<p align="center">
  <img src="Screenshot 2024-01-10 181922.png">
</p>

employee_df.info()
<p align="center">
  <img src="Screenshot 2024-01-10 181859.png">
</p>



employee_df.describe()

<p align="center">
  <img src="Screenshot 2024-01-10 195542.png">
</p>

## III-DATA VISUALIZATIONS
### 1) DATA CLEANING

Let's replace the 'Attritition' column with integers before performing any visualizations

<p align="center">
  <img src="Screenshot 2024-01-10 195959.png">
</p>

Right now Let's replace the 'Attritition' column with integers before performing any visualizations

<p align="center">
  <img src="Screenshot 2024-01-10 195959.png">
</p>

### 2) MISSING DATA
-LIBRARY SEABORN

Let's see if we have any missing data

<p align="center">
  <img src="Screenshot 2024-01-10 201558.png">
</p>

Fortunately there is no missing data in this case!

### -VISUALISATION (HISTOGRAM)

<p align="center">
  <img src="Screenshot 2024-01-10 201858.png">
</p>


Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavy
We are going to drop 'EmployeeCount' and 'Standardhours' because they do not change from one employee to the other


employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)

Let's see how many employees left the company!

left_df        = employee_df[employee_df['Attrition'] == 1]
stayed_df      = employee_df[employee_df['Attrition'] == 0]

print("Total =", len(employee_df))

print("Number of employees who left the company =", len(left_df))
print("Percentage of employees who left the company =", 1.*len(left_df)/len(employee_df)*100.0, "%")

print("Number of employees who did not leave the company (stayed) =", len(stayed_df))

print("Percentage of employees who did not leave the company (stayed) =", 1.*len(stayed_df)/len(employee_df)*100.0, "%")

Total = 1470

Number of employees who left the company = 237

Percentage of employees who left the company = 16.122448979591837 %

Number of employees who did not leave the company (stayed) = 1233

Percentage of employees who did not leave the company (stayed) = 83.87755102040816 %

left_df.describe()
<p align="center">
  <img src="Screenshot 2024-01-10 201944.png">
</p>

stayed_df.describe()
<p align="center">
  <img src="Screenshot 2024-01-10 202018.png">
</p>
correlations = employee_df.corr()
f, ax = plt.subplots(figsize = (20, 25))
sns.heatmap(correlations, annot = True)
<p align="center">
  <img src="Screenshot 2024-01-10 202211.png">
</p>
<p align="center">
  <img src="Screenshot 2024-01-10 202143.png">
</p>


###-RELATIONSHIP BETWEEN AGE AND ATTRITION

plt.figure(figsize=[25, 12])
sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df)
<p align="center">
  <img src="Screenshot 2024-01-10 202438.png">
</p>

plt.figure(figsize=[20,20])

plt.subplot(411)

sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)

plt.subplot(412)

sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df)

plt.subplot(413)

sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df)

plt.subplot(414)

sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)

<p align="center">
  <img src="Screenshot 2024-01-10 181620.png">
</p>
<p align="center">
  <img src="Screenshot 2024-01-10 181659.png">
</p>
KDE (kernel Density Estimate) PLOT [kdeplot]


plt.figure(figsize = (12,7))

sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'g')

plt.xlabel('Distance From Home')
<p align="center">
  <img src="Screenshot 2024-01-10 181506.png">
</p>
 

plt.figure(figsize=(12,7))

sns.kdeplot(left_df['YearsWithCurrManager'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label = 'Employees who Stayed', shade = True, color = 'g')

plt.xlabel('Years With Current Manager')
<p align="center">
  <img src="Screenshot 2024-01-10 181450.png">
</p>


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['TotalWorkingYears'], shade = True, label = 'Employees who left', color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'], shade = True, label = 'Employees who Stayed', color = 'g')

plt.xlabel('Total Working Years')


<p align="center">
  <img src="Screenshot 2024-01-10 181429.png">
</p>

plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data = employee_df)

<p align="center">
  <img src="Screenshot 2024-01-10 181409.png">
</p>

Let's see the monthly income vs. job role


plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)
<p align="center">
  <img src="Screenshot 2024-01-10 181346.png">
</p>


## IV) CREATE TESTING AND TRAINING DATASET & PERFORM DATA CLEANING

employee_df.head(3)

<p align="center">
  <img src="Screenshot 2024-01-10 215017.png">
</p>

X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_cat

<p align="center">
  <img src="Screenshot 2024-01-10 215035.png">
</p>

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()

X_cat.shape

(1470, 26)

X_cat = pd.DataFrame(X_cat)
X_cat

<p align="center">
  <img src="Screenshot 2024-01-10 181255.png">
</p>

<p align="center">
  <img src="Screenshot 2024-01-10 214819.png">
</p>

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#scaler.fit(X_all)

X_all.columns = X_all.columns.astype(str)

X = scaler.fit_transform(X_all)


X
<p align="center">
  <img src="Screenshot 2024-01-10 181147.png">
</p>


y = employee_df['Attrition']


y


<p align="center">
  <img src="Screenshot 2024-01-10 223907.png">
</p>


## V) TRAIN AND EVALUATE A LOGISTIC REGRESSION CLASSIFIER

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train.shape

(1102, 50)

X_test.shape

(368, 50)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


y_pred

<p align="center">
  <img src="Screenshot 2024-01-10 181126.png">
</p>


from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


Accuracy 87.77173913043478 %
Testing Set Performance

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)

<p align="center">
  <img src="Screenshot 2024-01-10 181055.png">
</p>

print(classification_report(y_test, y_pred))

              precision    recall  f1-score   support

           0       0.90      0.97      0.93       317
           1       0.62      0.31      0.42        51

    accuracy                           0.88       368
   macro avg       0.76      0.64      0.67       368
weighted avg       0.86      0.88      0.86       368

## VI) TRAIN AND EVALUATE A RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

Testing Set Performance

<p align="center">
  <img src="Screenshot 2024-01-10 181037.png">
</p>

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))

              precision    recall  f1-score   support

           0       0.88      0.99      0.93       317
           1       0.82      0.18      0.29        51

    accuracy                           0.88       368
   macro avg       0.85      0.59      0.61       368
weighted avg       0.87      0.88      0.85       368

## VI) TRAIN AND EVALUATE A DEEP LEARNING MODEL

import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50, )))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


model.summary()

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 500)               25500     
                                                                 
 dense_1 (Dense)             (None, 500)               250500    
                                                                 
 dense_2 (Dense)             (None, 500)               250500    
                                                                 
 dense_3 (Dense)             (None, 1)                 501       
                                                                 
=================================================================
Total params: 527001 (2.01 MB)
Trainable params: 527001 (2.01 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50)

Epoch 1/100
23/23 [==============================] - 2s 17ms/step - loss: 0.4510 - accuracy: 0.8176
Epoch 2/100
23/23 [==============================] - 0s 15ms/step - loss: 0.3646 - accuracy: 0.8485
Epoch 3/100
23/23 [==============================] - 0s 10ms/step - loss: 0.3189 - accuracy: 0.8757
Epoch 4/100
23/23 [==============================] - 0s 10ms/step - loss: 0.2977 - accuracy: 0.8911
Epoch 5/100
23/23 [==============================] - 0s 10ms/step - loss: 0.2869 - accuracy: 0.8893
Epoch 6/100
23/23 [==============================] - 0s 9ms/step - loss: 0.2378 - accuracy: 0.9111
Epoch 7/100
23/23 [==============================] - 0s 10ms/step - loss: 0.2019 - accuracy: 0.9265
Epoch 8/100
23/23 [==============================] - 0s 9ms/step - loss: 0.2235 - accuracy: 0.9138
Epoch 9/100
23/23 [==============================] - 0s 10ms/step - loss: 0.1800 - accuracy: 0.9247
Epoch 10/100
23/23 [==============================] - 0s 10ms/step - loss: 0.1327 - accuracy: 0.9501
Epoch 11/100
23/23 [==============================] - 0s 10ms/step - loss: 0.1170 - accuracy: 0.9601
Epoch 12/100
23/23 [==============================] - 0s 9ms/step - loss: 0.0872 - accuracy: 0.9682
Epoch 13/100
23/23 [==============================] - 0s 9ms/step - loss: 0.0587 - accuracy: 0.9746
Epoch 14/100
23/23 [==============================] - 0s 9ms/step - loss: 0.0512 - accuracy: 0.9837
Epoch 15/100
23/23 [==============================] - 0s 10ms/step - loss: 0.0240 - accuracy: 0.9936
Epoch 16/100
23/23 [==============================] - 0s 9ms/step - loss: 0.0971 - accuracy: 0.9637
Epoch 17/100
23/23 [==============================] - 0s 9ms/step - loss: 0.0888 - accuracy: 0.9619
Epoch 18/100
23/23 [==============================] - 0s 10ms/step - loss: 0.0440 - accuracy: 0.9891
Epoch 19/100
23/23 [==============================] - 0s 10ms/step - loss: 0.0175 - accuracy: 0.9955
Epoch 20/100
23/23 [==============================] - 0s 10ms/step - loss: 0.0056 - accuracy: 1.0000
Epoch 21/100
23/23 [==============================] - 0s 10ms/step - loss: 0.0028 - accuracy: 1.0000
Epoch 22/100
23/23 [==============================] - 0s 10ms/step - loss: 0.0016 - accuracy: 1.0000
Epoch 23/100
23/23 [==============================] - 0s 9ms/step - loss: 0.0010 - accuracy: 1.0000
Epoch 24/100
23/23 [==============================] - 0s 9ms/step - loss: 6.6903e-04 - accuracy: 1.0000
Epoch 25/100
23/23 [==============================] - 0s 10ms/step - loss: 5.2216e-04 - accuracy: 1.0000
Epoch 26/100
23/23 [==============================] - 0s 9ms/step - loss: 3.7797e-04 - accuracy: 1.0000
Epoch 27/100
23/23 [==============================] - 0s 9ms/step - loss: 3.0655e-04 - accuracy: 1.0000
Epoch 28/100
23/23 [==============================] - 0s 10ms/step - loss: 2.5651e-04 - accuracy: 1.0000
Epoch 29/100
23/23 [==============================] - 0s 10ms/step - loss: 2.0452e-04 - accuracy: 1.0000
Epoch 30/100
23/23 [==============================] - 0s 10ms/step - loss: 1.7081e-04 - accuracy: 1.0000
Epoch 31/100
23/23 [==============================] - 0s 10ms/step - loss: 1.4418e-04 - accuracy: 1.0000
Epoch 32/100
23/23 [==============================] - 0s 9ms/step - loss: 1.2326e-04 - accuracy: 1.0000
Epoch 33/100
23/23 [==============================] - 0s 10ms/step - loss: 1.0431e-04 - accuracy: 1.0000
Epoch 34/100
23/23 [==============================] - 0s 10ms/step - loss: 1.1086e-04 - accuracy: 1.0000
Epoch 35/100
23/23 [==============================] - 0s 10ms/step - loss: 8.8137e-05 - accuracy: 1.0000
Epoch 36/100
23/23 [==============================] - 0s 10ms/step - loss: 6.8153e-05 - accuracy: 1.0000
Epoch 37/100
23/23 [==============================] - 0s 10ms/step - loss: 5.9136e-05 - accuracy: 1.0000
Epoch 38/100
23/23 [==============================] - 0s 11ms/step - loss: 5.2380e-05 - accuracy: 1.0000
Epoch 39/100
23/23 [==============================] - 0s 11ms/step - loss: 4.6318e-05 - accuracy: 1.0000
Epoch 40/100
23/23 [==============================] - 0s 10ms/step - loss: 4.1189e-05 - accuracy: 1.0000
Epoch 41/100
23/23 [==============================] - 0s 10ms/step - loss: 3.7144e-05 - accuracy: 1.0000
Epoch 42/100
23/23 [==============================] - 0s 11ms/step - loss: 3.3546e-05 - accuracy: 1.0000
Epoch 43/100
23/23 [==============================] - 0s 10ms/step - loss: 3.4924e-05 - accuracy: 1.0000
Epoch 44/100
23/23 [==============================] - 0s 10ms/step - loss: 2.7929e-05 - accuracy: 1.0000
Epoch 45/100
23/23 [==============================] - 0s 14ms/step - loss: 2.4028e-05 - accuracy: 1.0000
Epoch 46/100
23/23 [==============================] - 0s 16ms/step - loss: 2.2325e-05 - accuracy: 1.0000
Epoch 47/100
23/23 [==============================] - 0s 15ms/step - loss: 1.8983e-05 - accuracy: 1.0000
Epoch 48/100
23/23 [==============================] - 0s 15ms/step - loss: 1.8635e-05 - accuracy: 1.0000
Epoch 49/100
23/23 [==============================] - 0s 15ms/step - loss: 1.5692e-05 - accuracy: 1.0000
Epoch 50/100
23/23 [==============================] - 0s 15ms/step - loss: 1.3894e-05 - accuracy: 1.0000
Epoch 51/100
23/23 [==============================] - 0s 15ms/step - loss: 1.2775e-05 - accuracy: 1.0000
Epoch 52/100
23/23 [==============================] - 0s 17ms/step - loss: 1.1837e-05 - accuracy: 1.0000
Epoch 53/100
23/23 [==============================] - 0s 14ms/step - loss: 1.0768e-05 - accuracy: 1.0000
Epoch 54/100
23/23 [==============================] - 0s 14ms/step - loss: 1.0008e-05 - accuracy: 1.0000
Epoch 55/100
23/23 [==============================] - 0s 16ms/step - loss: 9.2207e-06 - accuracy: 1.0000
Epoch 56/100
23/23 [==============================] - 0s 15ms/step - loss: 8.6202e-06 - accuracy: 1.0000
Epoch 57/100
23/23 [==============================] - 0s 15ms/step - loss: 8.0610e-06 - accuracy: 1.0000
Epoch 58/100
23/23 [==============================] - 0s 16ms/step - loss: 7.5208e-06 - accuracy: 1.0000
Epoch 59/100
23/23 [==============================] - 0s 15ms/step - loss: 6.9780e-06 - accuracy: 1.0000
Epoch 60/100
23/23 [==============================] - 0s 15ms/step - loss: 6.5159e-06 - accuracy: 1.0000
Epoch 61/100
23/23 [==============================] - 0s 15ms/step - loss: 6.1220e-06 - accuracy: 1.0000
Epoch 62/100
23/23 [==============================] - 0s 14ms/step - loss: 5.7757e-06 - accuracy: 1.0000
Epoch 63/100
23/23 [==============================] - 0s 15ms/step - loss: 5.4197e-06 - accuracy: 1.0000
Epoch 64/100
23/23 [==============================] - 0s 15ms/step - loss: 5.0920e-06 - accuracy: 1.0000
Epoch 65/100
23/23 [==============================] - 0s 14ms/step - loss: 4.8311e-06 - accuracy: 1.0000
Epoch 66/100
23/23 [==============================] - 0s 15ms/step - loss: 4.5565e-06 - accuracy: 1.0000
Epoch 67/100
23/23 [==============================] - 0s 11ms/step - loss: 4.3306e-06 - accuracy: 1.0000
Epoch 68/100
23/23 [==============================] - 0s 9ms/step - loss: 4.0937e-06 - accuracy: 1.0000
Epoch 69/100
23/23 [==============================] - 0s 9ms/step - loss: 3.8956e-06 - accuracy: 1.0000
Epoch 70/100
23/23 [==============================] - 0s 10ms/step - loss: 3.6959e-06 - accuracy: 1.0000
Epoch 71/100
23/23 [==============================] - 0s 11ms/step - loss: 3.5148e-06 - accuracy: 1.0000
Epoch 72/100
23/23 [==============================] - 0s 10ms/step - loss: 3.3709e-06 - accuracy: 1.0000
Epoch 73/100
23/23 [==============================] - 0s 10ms/step - loss: 3.2051e-06 - accuracy: 1.0000
Epoch 74/100
23/23 [==============================] - 0s 10ms/step - loss: 3.0641e-06 - accuracy: 1.0000
Epoch 75/100
23/23 [==============================] - 0s 10ms/step - loss: 2.9230e-06 - accuracy: 1.0000
Epoch 76/100
23/23 [==============================] - 0s 10ms/step - loss: 2.7888e-06 - accuracy: 1.0000
Epoch 77/100
23/23 [==============================] - 0s 10ms/step - loss: 2.6744e-06 - accuracy: 1.0000
Epoch 78/100
23/23 [==============================] - 0s 10ms/step - loss: 2.5800e-06 - accuracy: 1.0000
Epoch 79/100
23/23 [==============================] - 0s 10ms/step - loss: 2.4469e-06 - accuracy: 1.0000
Epoch 80/100
23/23 [==============================] - 0s 11ms/step - loss: 2.3608e-06 - accuracy: 1.0000
Epoch 81/100
23/23 [==============================] - 0s 10ms/step - loss: 2.3016e-06 - accuracy: 1.0000
Epoch 82/100
23/23 [==============================] - 0s 10ms/step - loss: 2.1690e-06 - accuracy: 1.0000
Epoch 83/100
23/23 [==============================] - 0s 11ms/step - loss: 2.0898e-06 - accuracy: 1.0000
Epoch 84/100
23/23 [==============================] - 0s 11ms/step - loss: 2.0108e-06 - accuracy: 1.0000
Epoch 85/100
23/23 [==============================] - 0s 10ms/step - loss: 1.9414e-06 - accuracy: 1.0000
Epoch 86/100
23/23 [==============================] - 0s 10ms/step - loss: 1.8623e-06 - accuracy: 1.0000
Epoch 87/100
23/23 [==============================] - 0s 10ms/step - loss: 1.7979e-06 - accuracy: 1.0000
Epoch 88/100
23/23 [==============================] - 0s 11ms/step - loss: 1.7343e-06 - accuracy: 1.0000
Epoch 89/100
23/23 [==============================] - 0s 10ms/step - loss: 1.6732e-06 - accuracy: 1.0000
Epoch 90/100
23/23 [==============================] - 0s 11ms/step - loss: 1.6086e-06 - accuracy: 1.0000
Epoch 91/100
23/23 [==============================] - 0s 11ms/step - loss: 1.5592e-06 - accuracy: 1.0000
Epoch 92/100
23/23 [==============================] - 0s 12ms/step - loss: 1.4922e-06 - accuracy: 1.0000
Epoch 93/100
23/23 [==============================] - 0s 10ms/step - loss: 1.4434e-06 - accuracy: 1.0000
Epoch 94/100
23/23 [==============================] - 0s 11ms/step - loss: 1.3954e-06 - accuracy: 1.0000
Epoch 95/100
23/23 [==============================] - 0s 10ms/step - loss: 1.3572e-06 - accuracy: 1.0000
Epoch 96/100
23/23 [==============================] - 0s 12ms/step - loss: 1.3092e-06 - accuracy: 1.0000
Epoch 97/100
23/23 [==============================] - 0s 10ms/step - loss: 1.2623e-06 - accuracy: 1.0000
Epoch 98/100
23/23 [==============================] - 0s 11ms/step - loss: 1.2239e-06 - accuracy: 1.0000
Epoch 99/100
23/23 [==============================] - 0s 10ms/step - loss: 1.1866e-06 - accuracy: 1.0000
Epoch 100/100
23/23 [==============================] - 0s 11ms/step - loss: 1.1477e-06 - accuracy: 1.0000

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


12/12 [==============================] - 0s 3ms/step
[ ]
  1
y_pred

array([[False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [ True],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [False],
       [ True],
       [False],
       [False],
       [False],
       [False]])

epochs_hist.history.keys()

dict_keys(['loss', 'accuracy'])

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])

<p align="center">
  <img src="Screenshot 2024-01-10 181008.png">
</p>

plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])

<p align="center">
  <img src="Screenshot 2024-01-10 180954.png">
</p>

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
# Testing Set Performance



print(classification_report(y_test, y_pred))

              precision    recall  f1-score   support

           0       0.90      0.93      0.91       317
           1       0.44      0.35      0.39        51

    accuracy                           0.85       368
   macro avg       0.67      0.64      0.65       368
weighted avg       0.84      0.85      0.84       368

Author
Dr Lebede Ngartera

Other Contributors
Kaggle Team.
