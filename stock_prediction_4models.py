#import libraries

#visualizing libraries
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import math

#analysing libraries
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#set the datae to retrive the data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2018, 1, 11)

df = web.DataReader('AAPL', 'yahoo', start, end)

#find the moving avarage
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

#plotting the mavg
#adjusting the size of matoplotlib
mpl.rc('figure', figsize=(8,7))
mpl.__version__

#adjusting the style of matplotlib to grid plot
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()

#return deviation to determine risk and deviation
rets = close_px / close_px.shift(1) -1
rets.plot(label='return')

#competetores stock
dfcomf = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo', start=start, end=end)['Adj Close']

#copetetores analysis
retscomp = dfcomf.pct_change()
corr = retscomp.corr()

plt.scatter(retscomp.AAPL, retscomp.GE)
plt.xlabel('Return AAPL')
plt.ylabel('Return GE')

pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10,10))

#heat map visualize
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)

#stock return rate risk
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected Returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
  plt.annotate(
      label,
      xy = (x, y), xytext = (20, -20),
      textcoords='offset points', ha = 'right', va = 'bottom',
      bbox = dict(boxstyle = 'round, pad=0.5', fc = 'yellow', alpha = 0.1),
      arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
  )

#predicting the stock
dfreg = df.loc[:, ['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

#pre processing and cross validation
# Drop Missing values
dfreg.fillna(value=-99999, inplace=True)

# we want to separate the 1 percent of the data to the forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluatio
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
print('Dimension of X',X.shape)
print('Dimension of y',y.shape)

#splitting the data to test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Building the models

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

#knn
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

#evaluation
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

print('Confidence of linear regression : ', confidencereg)
print('Confidence of linear quadratic regression : ', confidencepoly2)
print('Confidence of linear quadratic regression : ', confidencepoly3)
print('Confidence of linear KNN : ', confidenceknn)

#prediction of best fit regression model
forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan
print("\n set:\n {0}\n\n conf:\n {1}\n\n out: {2}\n\n".format(forecast_set, confidencereg, forecast_out))

#plotting prediction
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
