# NBA MVP Prediction Modeling

The 2022-2023 NBA season witnessed a highly contested MVP race that captivated fans and analysts alike. Nikola Jokic, Joel Embiid, and Giannis Antetokounmpo were at the heart of the competition, all with a captivating narrative and compelling case to make. In what was easily the most heated MVP debate since 2017, topics like racial bias, voter fatigue and the historical context of the MVP award were all brought into question. 

In the face of such a highly contested race, I seeked an objective, data-driven approach to determining the MVP. Utilizing machine learning and statistical modeling, I set out to create a prediction model that leverages historical voter data to establish a consistent rubric on evaluating the Most Valuable Player


```python
import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from datetime import date
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import xgboost as xgb
from functools import partial

from nba_mvp_predictor_helper_functions import *

notebook_path = os.path.abspath("")
os.chdir(notebook_path)

pd.options.mode.chained_assignment = None  # default='warn'
```

## Utilizing Past MVP Voting Data

While voter behavior in the past has never been consistent, it does offer us insight into what voters value. I started by using MVP results data from the past decade 2013 - 2022 (scraped from basketball-reference.com - see data scraper for more information). 

I decided that the output I wanted to predict was Vote Share, which is the % of Points a player received out of the total maximum points available. I liked this metric over the actual place a player a came in b/c it provided contextual numerical distance and was already scaled between 0 and 1. Additionally, I decided to filter for only the top 5 vote getters from each season as my training set.


```python
mvp_df = pd.read_csv('Data/mvp_results_2013_2022.csv')
mvp_df = (mvp_df.groupby('Season').apply(lambda x: x.nlargest(5, 'Pts Won')).reset_index(drop=True))
mvp_df.head()
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Age</th>
      <th>Season</th>
      <th>Tm</th>
      <th>First</th>
      <th>Pts Won</th>
      <th>Pts Max</th>
      <th>Share</th>
      <th>G</th>
      <th>MP</th>
      <th>PTS</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>FG%</th>
      <th>3P%</th>
      <th>FT%</th>
      <th>WS</th>
      <th>WS/48</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LeBron James</td>
      <td>28</td>
      <td>2013</td>
      <td>MIA</td>
      <td>120.0</td>
      <td>1207.0</td>
      <td>1210.0</td>
      <td>0.998</td>
      <td>76.0</td>
      <td>37.9</td>
      <td>26.8</td>
      <td>8.0</td>
      <td>7.3</td>
      <td>1.7</td>
      <td>0.9</td>
      <td>0.565</td>
      <td>0.406</td>
      <td>0.753</td>
      <td>19.3</td>
      <td>0.322</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Carmelo Anthony</td>
      <td>28</td>
      <td>2013</td>
      <td>NYK</td>
      <td>1.0</td>
      <td>475.0</td>
      <td>1210.0</td>
      <td>0.393</td>
      <td>67.0</td>
      <td>37.0</td>
      <td>28.7</td>
      <td>6.9</td>
      <td>2.6</td>
      <td>0.8</td>
      <td>0.5</td>
      <td>0.449</td>
      <td>0.379</td>
      <td>0.830</td>
      <td>9.5</td>
      <td>0.184</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kevin Durant</td>
      <td>25</td>
      <td>2014</td>
      <td>OKC</td>
      <td>119.0</td>
      <td>1232.0</td>
      <td>1250.0</td>
      <td>0.986</td>
      <td>81.0</td>
      <td>38.5</td>
      <td>32.0</td>
      <td>7.4</td>
      <td>5.5</td>
      <td>1.3</td>
      <td>0.7</td>
      <td>0.503</td>
      <td>0.391</td>
      <td>0.873</td>
      <td>19.2</td>
      <td>0.295</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LeBron James</td>
      <td>29</td>
      <td>2014</td>
      <td>MIA</td>
      <td>6.0</td>
      <td>891.0</td>
      <td>1250.0</td>
      <td>0.713</td>
      <td>77.0</td>
      <td>37.7</td>
      <td>27.1</td>
      <td>6.9</td>
      <td>6.3</td>
      <td>1.6</td>
      <td>0.3</td>
      <td>0.567</td>
      <td>0.379</td>
      <td>0.750</td>
      <td>15.9</td>
      <td>0.264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Stephen Curry</td>
      <td>26</td>
      <td>2015</td>
      <td>GSW</td>
      <td>100.0</td>
      <td>1198.0</td>
      <td>1300.0</td>
      <td>0.922</td>
      <td>80.0</td>
      <td>32.7</td>
      <td>23.8</td>
      <td>4.3</td>
      <td>7.7</td>
      <td>2.0</td>
      <td>0.2</td>
      <td>0.487</td>
      <td>0.443</td>
      <td>0.914</td>
      <td>15.7</td>
      <td>0.288</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Selection & Engineering

Next I needed to select for the features that the model would use. They are categorized as follows:
- Scaled Counting Statistics: I used a min-max scaler on the traditional countings stats to ensures that all features have a similar influence on the model and prevents features with larger values from dominating the model. I also seperated the min-max scaling to be done per season.
    - Points, Assists, Rebounds, Turnovers and Makes and Percentages for each type of shot (2P, 3P, and FT)
- Availability Stats as Percentage of Total: similar to the scaled counting stats, I took these as the % of total (82 games and 48 minutes)
    - Games, and Minutes Played
- Advanced Statistics: contains most of the frequently used advanced statistics. decided to split offensive and defensive stats b/c these defensive metrics were too biased towards big men, and adding them together would remove the model's ability to differentiate the importance of the two
    - PER, True Shooting %, Rebound, Assist and Turnover Percentage, Usage Rate, Off and Def Win Shares, and Off and Def Box Plus Minus
- Engineered Statistic: I decided to create a new statistic that captures the player's contribution to a team wins. it's similar Win Shares, but is not as dependent on efficiency
    - Win Contribution calculated as (Team Wins * Minutes Played / 48 * Games Played / 82 * Usage Rate -> then min-max scaled)


```python
#Stats_Df scraped using the NBA Scraper Notebook
stats_df = pd.read_csv('Data/season_stats_13_22.csv')
stats_df = df_transform(stats_df)
#Join to MVP DataFrame
mvp_train = join_dataframes(mvp_df, stats_df)
mvp_train.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Age</th>
      <th>Season</th>
      <th>Tm</th>
      <th>Actual_Rank</th>
      <th>First</th>
      <th>Share</th>
      <th>G</th>
      <th>3P</th>
      <th>3P%</th>
      <th>...</th>
      <th>TRB%</th>
      <th>AST%</th>
      <th>TOV%</th>
      <th>USG%</th>
      <th>OWS</th>
      <th>DWS</th>
      <th>OBPM</th>
      <th>DBPM</th>
      <th>Win_Contrib</th>
      <th>Minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LeBron James</td>
      <td>28</td>
      <td>2013</td>
      <td>MIA</td>
      <td>1.0</td>
      <td>120.0</td>
      <td>0.998</td>
      <td>0.93</td>
      <td>0.26</td>
      <td>0.41</td>
      <td>...</td>
      <td>13.1</td>
      <td>36.4</td>
      <td>12.4</td>
      <td>0.30</td>
      <td>14.6</td>
      <td>4.7</td>
      <td>9.3</td>
      <td>2.4</td>
      <td>0.90</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Carmelo Anthony</td>
      <td>28</td>
      <td>2013</td>
      <td>NYK</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.393</td>
      <td>0.82</td>
      <td>0.43</td>
      <td>0.38</td>
      <td>...</td>
      <td>10.8</td>
      <td>14.1</td>
      <td>9.3</td>
      <td>0.36</td>
      <td>7.5</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>-1.7</td>
      <td>0.74</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kevin Durant</td>
      <td>25</td>
      <td>2014</td>
      <td>OKC</td>
      <td>1.0</td>
      <td>119.0</td>
      <td>0.986</td>
      <td>0.99</td>
      <td>0.45</td>
      <td>0.39</td>
      <td>...</td>
      <td>10.8</td>
      <td>26.7</td>
      <td>12.2</td>
      <td>0.33</td>
      <td>14.8</td>
      <td>4.4</td>
      <td>8.8</td>
      <td>1.4</td>
      <td>0.94</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LeBron James</td>
      <td>29</td>
      <td>2014</td>
      <td>MIA</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.713</td>
      <td>0.94</td>
      <td>0.28</td>
      <td>0.38</td>
      <td>...</td>
      <td>11.5</td>
      <td>32.0</td>
      <td>14.4</td>
      <td>0.31</td>
      <td>12.3</td>
      <td>3.7</td>
      <td>7.8</td>
      <td>1.1</td>
      <td>0.75</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Stephen Curry</td>
      <td>26</td>
      <td>2015</td>
      <td>GSW</td>
      <td>1.0</td>
      <td>100.0</td>
      <td>0.922</td>
      <td>0.98</td>
      <td>0.68</td>
      <td>0.44</td>
      <td>...</td>
      <td>7.0</td>
      <td>38.6</td>
      <td>14.3</td>
      <td>0.29</td>
      <td>11.5</td>
      <td>4.1</td>
      <td>8.2</td>
      <td>1.7</td>
      <td>0.79</td>
      <td>0.68</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
#Generate Correlation Matrix
corr_matrix = mvp_train.corr()

# Plot correlation matrix
plt.figure(figsize=(10,8))
plt.title("Correlation Matrix")
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()
```

    /var/folders/g9/lxmtwlzn7rb5ds2ctlg85t480000gn/T/ipykernel_50186/3787025150.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      corr_matrix = mvp_train.corr()



    
![png](output_6_1.png)
    


## Model Training and Evaluation

As I started to train the model, I realized some of standard functions traditionally used in Machine Learning modeling problems will not work for this particular problem. I decided to write custom functions for the following:

1. train-test-split: instead of randomly splitting rows into training and test sets, I decided to randomly split entire seasons of data. because there's limited training data compared to traditional ML problems, it was important that each set had the proportion of MVP winners
2. weighted error: while the model was predicting "share", the question we're most interested in is obviously who won the MVP, which is the rank of the actual share. a standard root mean square error isn't as helpful here so I used a custom error formula that weighs the difference in the predicted rank vs. actual rank such that
    - If the actual rank is 1 (aka the real MVP), the absolute difference between the predicted rank will be multiplied by 5
    - If the actual rank is 2 (runner-up), the abs. difference will be multiplied by a weight of 3
    - All other actual rankis, the weights will be 1


```python
X, y = xy_split(mvp_train)
```


```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test, train, test = season_train_test_split(mvp_train, 4, random_state = 42)
```

### 1. Linear Regression
Starting with a simple linear regression model, and found positive initial results where the test data had 0 Weighted error (meaning it accurately predicted all of the MVP ranks). While this is promising, it could mean the model is overfitting)


```python
# create a LinearRegression model
linear_reg = LinearRegression()

# train the LR model on the training data
linear_reg.fit(X_train, y_train)

# make predictions on the test set
y_pred = linear_reg.predict(X_test)

#Evaluate the model
lr_rmse = mean_squared_error(y_test, y_pred, squared=False).round(2)
print('Linear Regression RMSE:', lr_rmse)
lr_weighted_error = weighted_error(y_test, y_pred, test)
print(f"Weighted Rank Error: {lr_weighted_error}")
```

    Linear Regression RMSE: 0.25
    Weighted Rank Error: 0.0


### 2. Ridge Regression
Next I wanted to try a Ridge Regression model that would regularize the colinearity of the feature variables, which is obviously a major issue with a lot of these advanced metrics


```python
# Initialize the model
ridge = Ridge(alpha=0.5)

# Fit the model on the training data
ridge.fit(X_train, y_train)

# Predict on the test data
y_pred = ridge.predict(X_test)

# Calculate Errors
r_rmse = mean_squared_error(y_test, y_pred, squared=False).round(2)
print('Ridge Regression RMSE:', r_rmse)
r_weightd_error = weighted_error(y_test, y_pred, test)
print(f"Weighted Rank Error: {r_weightd_error}")
```

    Ridge Regression RMSE: 0.19
    Weighted Rank Error: 0.0


### 3. XgBoost

For XGBoost, I wanted to apply more statistical rigor. I re-split the data now to include a validation set, and applied the weighted error function as a custom evaluation metric that would train the model.


```python
# Add Validation Set for XgBoost (8-1-1 split)
X_train, X_valid, X_test, y_train, y_valid, y_test, train, valid, test = season_train_test_split(mvp_train, 4, validation=True, random_state = 42)
```


```python
# Prepare the data
train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)

# Set the parameters for the XGBoost model
params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_jobs': -1,
}

#Train the data
num_rounds = 1000
watchlist = [(train_data, 'train'), (valid_data, 'valid')]

model = xgb.train(
    params,
    train_data,
    num_rounds,
    watchlist,
    early_stopping_rounds=50,
    obj=custom_obj,  # Use the custom objective function
    feval=None,  # Temporarily set feval to None
    maximize=False,
    verbose_eval=10,
)

# Now that the model is trained, pass it to the custom_eval function using functools.partial
custom_eval_with_model = partial(custom_eval, model)

# Re-run the evaluation with the custom_eval function that includes the model
train_eval = custom_eval_with_model(dtrain=train_data, ref_df=train)
valid_eval = custom_eval_with_model(dtrain=valid_data, ref_df=valid)

print("Train evaluation:", train_eval)
print("Validation evaluation:", valid_eval)
```

    [0]	train-rmse:0.30410	valid-rmse:0.33725
    [10]	train-rmse:0.13610	valid-rmse:0.16009
    [20]	train-rmse:0.06457	valid-rmse:0.12204
    [30]	train-rmse:0.03358	valid-rmse:0.12882
    [40]	train-rmse:0.01881	valid-rmse:0.13018
    [50]	train-rmse:0.01107	valid-rmse:0.12999
    [60]	train-rmse:0.00664	valid-rmse:0.13001
    [68]	train-rmse:0.00444	valid-rmse:0.12971
    Train evaluation: ('weighted_error', 0.0)
    Validation evaluation: ('weighted_error', 2.0)


    /usr/local/Cellar/jupyterlab/3.4.8_1/libexec/lib/python3.11/site-packages/xgboost/core.py:617: FutureWarning: Pass `evals` as keyword args.
      warnings.warn(msg, FutureWarning)



```python
test_data = xgb.DMatrix(X_test)

# Make predictions on the test set
y_pred_test = model.predict(test_data)

# Calculate Errors
xgb_rmse = mean_squared_error(y_test, y_pred_test, squared=False).round(2)
print('XGBoost RMSE:', xgb_rmse)
xgb_weightd_error = weighted_error(y_test, y_pred_test, test)
print(f"Weighted Rank Error: {xgb_weightd_error}")
```

    XGBoost RMSE: 0.24
    Weighted Rank Error: 6.0



```python
#Show Results of the XGBoost model
mvp2_xgb = mvp_train.copy()
y_pred = model.predict(xgb.DMatrix(X))
mvp2_xgb['Pred_Share'] = y_pred
mvp2_xgb['Predicted_Rank'] = mvp2_xgb.groupby('Season')['Pred_Share'].rank(ascending=False, method='dense')
mvp2_xgb = mvp2_xgb[['Player', 'Season', 'Actual_Rank', 'Predicted_Rank']]
mvp2_xgb.loc[mvp2_xgb.Actual_Rank != mvp2_xgb.Predicted_Rank]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Season</th>
      <th>Actual_Rank</th>
      <th>Predicted_Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Kevin Durant</td>
      <td>2014</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LeBron James</td>
      <td>2014</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Russell Westbrook</td>
      <td>2017</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>James Harden</td>
      <td>2017</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Kawhi Leonard</td>
      <td>2017</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Joel Embiid</td>
      <td>2022</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Giannis Antetokounmpo</td>
      <td>2022</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



Despite being the most robust model, XGBoost actually had the highest Weighted Rank Error, and when we look at its predicted results from the past decade of data, it actually overturns 2 races. The 2014 race where it ranks LeBron over KD, and in 2017 where it ranks Kawhi over both Russ and Harden, which while suprising is not actually totally outlandish

## Model Evaluation


```python
# Aggregate Error Metrics for each Model
error_metrics = pd.DataFrame({
    'RMSE': [lr_rmse, r_rmse, xgb_rmse],
    'Weighted Rank Error': [lr_weighted_error, r_weightd_error, xgb_weightd_error]
}, index=['Linear Regression', 'Ridge Regression', 'XGBoost'])

error_metrics
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RMSE</th>
      <th>Weighted Rank Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear Regression</th>
      <td>0.25</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ridge Regression</th>
      <td>0.19</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>XGBoost</th>
      <td>0.24</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



## 2023 Prediction


```python
df_23 = pd.read_csv('Data/2023_data.csv')
df_23 = df_transform(df_23)
df_23 = df_23.loc[(df_23.G > 0.5) & (df_23.Minutes > 0.5)]
non_feature_cols = df_23.iloc[:, :3].columns
X_23 = df_23.drop(columns = non_feature_cols)
```


```python
#Linear Regression Prediction
lr_23 = df_23.copy()
lr_23['Pred_Share'] = linear_reg.predict(X_23).round(2)
#lr_23 = lr_23.loc[(lr_23.PER > 16) & (lr_23.G > .6)]
lr_23['Predicted_Rank'] = lr_23['Pred_Share'].rank(ascending=False, method='dense')
lr_23 = lr_23[['Player','Win_Contrib', 'PER', 'Pred_Share', 'Predicted_Rank']]
lr_23.sort_values(by = ['Predicted_Rank'], ascending = True).head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Win_Contrib</th>
      <th>PER</th>
      <th>Pred_Share</th>
      <th>Predicted_Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>399</th>
      <td>Nikola Jokić</td>
      <td>0.66</td>
      <td>31.5</td>
      <td>0.96</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>Joel Embiid</td>
      <td>0.90</td>
      <td>31.4</td>
      <td>0.43</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Luka Dončić</td>
      <td>0.67</td>
      <td>28.7</td>
      <td>0.42</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>220</th>
      <td>James Harden</td>
      <td>0.57</td>
      <td>21.6</td>
      <td>0.39</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>129</th>
      <td>Domantas Sabonis</td>
      <td>0.55</td>
      <td>23.5</td>
      <td>0.38</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Ridge Regression Prediction
rr_23 = df_23.copy()
rr_23['Pred_Share'] = ridge.predict(X_23).round(2)
#rr_23 = rr_23.loc[rr_23.Win_Contrib > 0]
rr_23['Predicted_Rank'] = rr_23['Pred_Share'].rank(ascending=False, method='dense')
rr_23 = rr_23[['Player','Win_Contrib', 'PER', 'Pred_Share', 'Predicted_Rank']]
rr_23.sort_values(by = ['Predicted_Rank'], ascending = True).head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Win_Contrib</th>
      <th>PER</th>
      <th>Pred_Share</th>
      <th>Predicted_Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>399</th>
      <td>Nikola Jokić</td>
      <td>0.66</td>
      <td>31.5</td>
      <td>0.93</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>Joel Embiid</td>
      <td>0.90</td>
      <td>31.4</td>
      <td>0.63</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Luka Dončić</td>
      <td>0.67</td>
      <td>28.7</td>
      <td>0.62</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Giannis Antetokounmpo</td>
      <td>0.90</td>
      <td>29.0</td>
      <td>0.53</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>238</th>
      <td>Jayson Tatum</td>
      <td>1.00</td>
      <td>23.7</td>
      <td>0.46</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#xGboost Prediction
xgb_23 = df_23.copy()
xgb_23['Pred_Share'] = model.predict(xgb.DMatrix(X_23)).round(2)
xgb_23['Predicted_Rank'] = xgb_23['Pred_Share'].rank(ascending=False, method='dense')
xgb_23 = xgb_23[['Player','Win_Contrib', 'PER', 'Pred_Share','Predicted_Rank']]
xgb_23.sort_values(by = ['Predicted_Rank'], ascending = True).head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Win_Contrib</th>
      <th>PER</th>
      <th>Pred_Share</th>
      <th>Predicted_Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>399</th>
      <td>Nikola Jokić</td>
      <td>0.66</td>
      <td>31.5</td>
      <td>0.93</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Giannis Antetokounmpo</td>
      <td>0.90</td>
      <td>29.0</td>
      <td>0.85</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>Joel Embiid</td>
      <td>0.90</td>
      <td>31.4</td>
      <td>0.77</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Anthony Davis</td>
      <td>0.46</td>
      <td>27.8</td>
      <td>0.72</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>235</th>
      <td>Jaylen Brown</td>
      <td>0.84</td>
      <td>19.1</td>
      <td>0.70</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

All 3 models predicted Nikola Jokic as the MVP despite him being not the Vegas favorite. I think this speaks to the analytical marvel of Jokic's season, as well as the model's inability to account for narrative and voter fatigue.

Despite the limitations of these models, I found this exercise to be a refreshing look at how much can be evaluated by purely looking at the numbers. 
