# Time Series Assignment #

 - Time Series and Regression Analysis using Yen futures data
 
## Starting off with our relevant libraries as always ##

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

![.](<a href="https://imgur.com/8isFZvN"><img src="https://i.imgur.com/8isFZvN.jpg" title="source: imgur.com" /></a>)

## Grabbing the Settle price by date ##

```python
yen_settle = yen_futures['Settle']
yen_settle.plot(figsize = (12,6))
```

![.](<a href="https://imgur.com/5qFUB9g"><img src="https://i.imgur.com/5qFUB9g.jpg" title="source: imgur.com" /></a>)

## Using the HP (Hodrick-Prescott) Flter lets get rid of the noise and focus on the trend ##

```python
s = "Python syntax highlighting"
print s
```

![.]()

## Pulling out the Settle, Noise and Trend for our data ##

```python
yen_nt = pd.DataFrame({'Settle':yen_settle.values, 'Date':yen_settle.index, 'noise':ts_noise.values, 'trend':ts_trend.values})
yen_nt = yen_nt.set_index('Date')
yen_nt.tail()
```

![.](<a href="https://imgur.com/xs0fatN"><img src="https://i.imgur.com/xs0fatN.jpg" title="source: imgur.com" /></a>)
 
 
 ## PLotting the settle vs. trend, we should fully expect to graphs. One the shows almost candlestick like behavior and one that is far easier to digest. ##

```python
settle_v_trend = pd.DataFrame(yen_nt, columns=['Settle', 'trend'])
settle_v_trend = settle_v_trend.loc['2015':'2019']
settle_v_trend.plot(figsize=(20, 9))
```

![.](<a href="https://imgur.com/op2TZPp"><img src="https://i.imgur.com/op2TZPp.jpg" title="source: imgur.com" /></a>)
 
 ## Forecasting the Settle Price using an ARIMA Model (Autoregressive Integrated Moving Average Model) ##
 - Below is the performance of the model, Pvalues are questionable using this model

![.]()
 
# Conclusions #

##Based on your time series analysis, would you buy the yen now?

Is the risk of the yen expected to increase or decrease?

Based on the model evaluation, would you feel confident in using these models for trading?

If I were to follow the GARCH model absolutely yes I would, however both ARIMA and ARMA would cause a rise in doubt about the potential performance of the yen. The P value of the GARCH model was excellent though so it would take a lot for me not to buy according to the GARCH model, that instills a lot of confidence. As you aim further and further out yes the risk would increase as the model up close does a great job. ##
