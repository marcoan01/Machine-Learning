import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/marcoan01/MLA/main/ale.csv"            #get the dataset
df = pd.read_csv(url)                                                           #read the data set
train, test = train_test_split(df, test_size = 0.2)                             #split between train and test
x = np.array(train.iloc[0:,0:3])
y = np.array(train.iloc[0:,3:])
x_test = np.array(test.iloc[0:, 0:3])
y_test = np.array(test.iloc[0:,3:])

lm = linear_model.LinearRegression()   # Linear regression                      #create a LinearRegression object
lm.fit(x, y)                                                                    #Train the data
print("The optimal weight values are: " + str(lm.coef_))                        #Output the weight values
y_predict = lm.predict(x_test)
print("\n\nPredicted set of values: " + str(y_predict))
print("\n\nActual set of values: " + str(np.array(y_test)))
#output the prediction
print("MSE value is: " + str(mean_squared_error(y_test, y_predict, multioutput='raw_values')))  #output the MSE