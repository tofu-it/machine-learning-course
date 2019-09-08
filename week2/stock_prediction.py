# Import necessary packages

from datetime import timedelta
from datetime import datetime 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
#%matplotlib inline
import math
import datetime, time
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import io


df = pd.read_csv('tesla.csv')


#Indexing the date
df.set_index('Date', inplace=True)


#making a copy of the original data, just in case we need a later reference to the unaltered data
original = df.copy()

#Checking how the data acually looks like (colums, dates, etc.)
print (df.tail())


#Picking the rows we gonna use for the prediction by considering relationships. I.e. Open and Closing Prices
#are a direct indication if the price went up or down, also the margin tells us a bit of voilatility for the day.
#Apparently, a simple linear regression would not do necessarily find out these relationships, so we also need
#to define them. Then we can use those relationships as features rather than just the plain prices.

# New column: Percent Voilatility (HL_PCT)is given by High - Low

# New Colum: Percentual Change (PCT_change)

#Note: Volume seems to be relevant for voilatility, so we keep that as well.


df = df[['Open', 'High','Low','Close','Adj Close','Volume',]]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

#Selecting the columns we only need and forgetting everything else
df=df[['Adj Close','HL_PCT','PCT_change','Volume']]

#Right now, we are using stock prices, and want to forecast "Adj Close" if possible. 
#To make the layout more flexible, we define that intention here, so we can easily 
#change it later for other purposes, withouth having to go to the rest of the code 
#and change all the columns there  
forecast_col = 'Adj Close'

#data = df['Adj Close'].values[:]
#just curious how much datasets we actually have, so just printing that out.
print(len(df))

#fill non existing colums with something
df.fillna(-99999, inplace=True)

#forcast only 10% of the whole dataframe
forecast_out = int(math.ceil(0.01*len(df)))

#W0e only have features above, so let's create a label now
#Value will be the 'Adj Close' value from 10 days (or what ever we have defined 
#as a 'forecast_out shift'). In other words, we copy the 'Adj Close' column to
#a new colum named 'label', and move this whole colum ten rows back. 

df['label'] = df[forecast_col].shift(-forecast_out)


#Note: Features are attributes of something, that we think, may cause the 
# given price in 10 days in the future


#checking on the result
#print (df.head())


#Defining our training datasets 
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])


#let's train, and test it with 20% (size=0.2) of the data

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#X_train and y_train is used to fit our classifier
#defining our first classifier, using basic LinearRegression (Ordinary Least Squares) 
clf = LinearRegression() #Note: Test the n_jobs argument for threading
#train it with X_train and y_train
clf.fit(X_train, y_train)


#'fit' is synonymous with 'train score'  is synonymous with 'test'
# so, a simple way to check out how accurate our prediction may be
accuracy = clf.score(X_test,y_test)
print (accuracy)
#Note to self: Accuracy is the squared error

#predict on the classifier (X), can be single value, or array (x_lately, 
#which should be 24 days in advance (the value of forecast_out))
forecast_set = clf.predict(X_lately)

#Appending the forecast data to the dataframe, including the appropriate time stamps

#New empty column with no data
df['Forecast_LR'] = np.nan

#creating additional rows for the predicted data
last_date = df.iloc[-1].name
nextday = datetime.datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)
for i in forecast_set:
    next_date = nextday
    nextday += timedelta(days=1)
    #print (nextday)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
#Checking if everything looks good
#print (df.tail()



# 2nd Method of linear regression: Lasso

clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)
print (accuracy)


#predicting 
forecast_set = clf.predict(X_lately)

#New empty column for the Lasso results
df['Forecast_Lasso'] = np.nan
df['Forecast_Lasso'][-forecast_out:]=forecast_set
df.tail()



# 3th training method of linear regression: Ridge
clf = Ridge()
clf.fit(X_train, y_train)


accuracy = clf.score(X_test,y_test)
print (accuracy)


#predicting 
forecast_set = clf.predict(X_lately)


#New empty column for the Ridge results
df['Forecast_Ridge'] = np.nan
df['Forecast_Ridge'][-forecast_out:]=forecast_set

#Show the end of the dataframe to see the actual and the predicted prices side by side
print (df.tail())

#Now everything together in a single chart, along with the original data (only the last 40 days).

df['Forecast_LR'][-40:].plot(figsize=(23,15))
df['Forecast_Lasso'][-40:].plot(figsize=(23,15))
df['Forecast_Ridge'][-40:].plot(figsize=(23,15))
original['Adj Close'][-40:].plot(figsize=(23,15))
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Price at Close')

plt.show()
