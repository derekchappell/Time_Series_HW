# Time Series Assignment #

 - Time Series and Regression Analysis using Yen futures data
 
## Starting off with (most of) our relevant libraries as always ##

```python

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
%matplotlib inline
```
 ## Return Forecasting: Read Historical Daily Yen Futures Data. Below we read our relevant csv file ##

```python
yen_futures = pd.read_csv(
    Path("yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()
```

[.](<a href="https://imgur.com/8isFZvN"><img src="https://i.imgur.com/8isFZvN.jpg" title="source: imgur.com" /></a>)

## Grabbing the Settle price by date ##

```python
yen_settle = yen_futures['Settle']
yen_settle.plot(figsize = (12,6))
```

[.](<a href="https://imgur.com/5qFUB9g"><img src="https://i.imgur.com/5qFUB9g.jpg" title="source: imgur.com" /></a>)

## Using the HP (Hodrick-Prescott) Flter lets get rid of the noise and focus on the trend ##

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

ts_noise, ts_trend = sm.tsa.filters.hpfilter(yen_futures['Settle'])
ts_trend.plot(figsize=(20, 9))
plt.show()
```

[.](<a href="https://imgur.com/xs0fatN"><img src="https://i.imgur.com/xs0fatN.jpg" title="source: imgur.com" /></a>)

## Pulling out the Settle, Noise and Trend for our data ##

```python
yen_nt = pd.DataFrame({'Settle':yen_settle.values, 'Date':yen_settle.index, 'noise':ts_noise.values, 'trend':ts_trend.values})
yen_nt = yen_nt.set_index('Date')
yen_nt.tail()
```

[.](<a href="https://imgur.com/QYcFBwd"><img src="https://i.imgur.com/QYcFBwd.jpg" title="source: imgur.com" /></a>)
 
 
 ## PLotting the settle vs. trend, we should fully expect to graphs. One the shows almost candlestick like behavior and one that is far easier to digest. ##

```python
settle_v_trend = pd.DataFrame(yen_nt, columns=['Settle', 'trend'])
settle_v_trend = settle_v_trend.loc['2015':'2019']
settle_v_trend.plot(figsize=(20, 9))
```

[.](<a href="https://imgur.com/op2TZPp"><img src="https://i.imgur.com/op2TZPp.jpg" title="source: imgur.com" /></a>)
 
 ## Forecasting the Settle Price using an ARIMA Model (Autoregressive Integrated Moving Average Model) ##
 - Below is the performance of the model, Pvalues are questionable using this model

[.](<a href="https://imgur.com/QPrluoj"><img src="https://i.imgur.com/QPrluoj.jpg" title="source: imgur.com" /></a>)

## Here is the trend our model predicts ##

[.](<a href="https://imgur.com/SH6Px2n"><img src="https://i.imgur.com/SH6Px2n.jpg" title="source: imgur.com" /></a>)
 
# Conclusions #

##Based on your time series analysis, would you buy the yen now?

Is the risk of the yen expected to increase or decrease?

Based on the model evaluation, would you feel confident in using these models for trading?

If I were to follow the GARCH model absolutely yes I would, however both ARIMA and ARMA would cause a rise in doubt about the potential performance of the yen. The P value of the GARCH model was excellent though so it would take a lot for me not to buy according to the GARCH model, that instills a lot of confidence. As you aim further and further out yes the risk would increase as the model up close does a great job. ##


# Linear Regression was the second related skill set this assignment is going to cover #

## Starting off with (most of) our relevant libraries as always ##

```python
import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline
```

## Here we begin the process of using SKLearn's linear regression model to predict furture Yen prices ##
 - reading our csv into a DataFrame and setting the date as the properly formated index

```python
yen_futures = pd.read_csv(
    Path("yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()
```

[.](<a href="https://imgur.com/3FjdJvf"><img src="https://i.imgur.com/3FjdJvf.jpg" title="source: imgur.com" /></a>)

## Here we are creating our settle prices percentage change to get an idea of how the yen has moved and adding that column to the end of our DataFrame ##

```python
returns_settle = (yen_futures[["Settle"]].pct_change() * 100)
returns_settle = returns_settle.replace(-np.inf, np.nan).dropna()
yen_futures['settle'] = returns_settle
yen_futures.head()
```

[.](<a href="https://imgur.com/rsnkNr2"><img src="https://i.imgur.com/rsnkNr2.jpg" title="source: imgur.com" /></a>)

## Lets create a test and train DataFrames by utilizing the time date index by year in this case '99 to '17 for training and '18 to '19 for testing ##

```python
train = yen_futures[:'2017']
test = yen_futures['2018':]

X_train = train['Lagged_Return'].to_frame()
y_train = train['settle']
X_test = test['Lagged_Return'].to_frame()
y_test = test['settle']
```
## On a visual level our first round is not instilling a terribly large amount of confidence in this model. ##

[.](<a href="https://imgur.com/OcMWLqn"><img src="https://i.imgur.com/OcMWLqn.jpg" title="source: imgur.com" /></a>)

## In Sample vs Out of Sample performance ##

```python
Out-of-Sample Root Mean Squared Error (RMSE): 0.41545437184712763
Out-of-Sample Root Mean Squared Error (RMSE): 0.5962037920929946
```

The RMSE of the out of sample predictions are a better fit for predicting what could happen to the price of the yen, however the plots of the data show the prediction inversing the actual data. Not reassuring on a visual level.
