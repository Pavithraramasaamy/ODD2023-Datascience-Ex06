# EX-06 FEATURE TRANSFORMATION

## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:

## Step1:
Read the given Data.

## Step2:
Clean the Data Set using Data Cleaning Process.

## Step3: Apply Feature Transformation techniques to all the features of the data set.

## Step4: Print the transformed features.

## PROGRAM:

```
Developed by: PAVITHRA R
Register Number: 212222230106\


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()


```

## OUTPUT:

## DATASET:


![1](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/8ac77c2d-2a8c-4e39-b021-3e4182fdc565)


## HEAD:

![2](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/88953517-1e97-4079-bd1f-7d3b64c25261)


## IS NULL:

![3](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/5a7812e6-a742-4cc4-8ec0-13c51e488e63)



## INFO:

![4](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/6b6e071d-16e8-4d08-b09b-48ca9e2bfc95)


## DESCRIBE:
![5](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/807e4ba7-4eed-488b-a012-8937442f3e82)


## BEFORE TRANSFORMATION:

![6](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/c8d9196b-9096-4933-8956-ed685d7c08e6)



![7](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/ee88e5e7-ba4a-4c6d-8c6c-fe2f6ca5a249)



![8](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/2da06791-4baf-4961-9df0-aed45f801990)



![9](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/66d3c04f-4cc6-4bc7-a710-dfe63f105ba7)


## LOG TRANSFORMATION:

![10](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/aa6d414e-840d-4a18-ac1d-8b004177a7e0)


## RECIPROCAL TRANSFORMATION:

![11](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/42e0c92d-58d7-42e4-b345-e05264d7ea73)


## SQUARE ROOT TRANSFORMATION:

![12](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/1ed8cc2f-3bca-4fec-a232-9cbb1bb179b9)



![13](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/f948669a-b0fe-48a1-a60e-9d4a86ce273d)

## POWER TRANSFORMER:

![Screenshot 2023-10-17 123604](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/9400295b-6123-408a-97de-8266f106188b)


## QUANTILE TRANSFORMER:

![15](https://github.com/Pavithraramasaamy/ODD2023-Datascience-Ex06/assets/118596964/ca1d5c09-f286-4d02-b020-142c4e12556f)


## RESULT:
  Thus the program has been executed successfully.
