# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot

2. Trace the best fit line and calculate the cost function

3. Calculate the gradient descent and plot the graph for it

4. Predict the profit for two population sizes.


## Program:
```python
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SACHIN.C
RegisterNumber:  212222230125
```

```python
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output:

### PLACEMENT DATA:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/10ef343d-67ea-448b-bc38-b63e4352748d)

### SALARY DATA:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/d4f17a03-2662-466e-bf80-325305e26724)

### CHECKING THE NULL() FUNCTION:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/1c25c117-d39b-412d-80ba-4f8a421f6b50)

### DATA DUPLICATE:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/ce8a2adc-79c1-4788-bb39-9d19e8fd75ed)

### PRINT DATA:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/88d185ee-7bdc-480a-ab8c-ebe838351be4)

### DATA STATUS:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/4404ca15-290a-4623-aa07-b32a7d07634b)

### Y_PREDCITION ARRAY:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/c102ee52-6942-4c11-b1f0-7b56c3a79fff)

### ACCURACY VALUE:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/877f5262-e7ae-458d-bb52-c3827b346cc8)

### CONFUSION ARRAY:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/464fe227-16ee-44d3-a4eb-08e054afcaf3)

### CLASSIFICATION REPORT:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/ade5c215-abbd-4348-919c-2d3a0ea204fe)

### PREDICTION OF LR:

![image](https://github.com/Sachin-vlr/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113497666/a705b308-847e-4cd7-be14-f167c022688b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
