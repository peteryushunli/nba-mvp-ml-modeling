import os
import random
from functools import partial
import time
import requests
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from scipy.spatial.distance import cdist
from datetime import date
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def xy_split(df):
    """
    Custom function to split the output (voter share) and the feature variables
    """
    
    y = df.Share

    non_feature_cols = df.iloc[:, :7].columns
    X = df.drop(columns = non_feature_cols)
    return X, y

def season_train_test_split(df, test_seasons, validation = False, random_state = 42):
    """
    This is a custom error function split the dataset into training and test data
    
    Inputs:
    df = dataframe of season stats and mvp voting results
    test_seasons = the number of seasons extracted as the test set (insert as an integer, even if validation is true)
    validation = binary flag to specify whether you want to have a validation set
    random_state = integer to control for the randomization process
    """
    
    #Select random seasons as test data
    start = df.Season.min()
    end = df.Season.max()
    
    #set Random State
    if random_state:
        random.seed(random_state)
        
    #Select random seasons for test and validation set    
    sample = random.sample(range(start, end+1), test_seasons)
    
    
    if validation == False:
        #Split data
        test = df[df.Season.isin(sample)]
        train = df[~df.Season.isin(sample)]

        #Split by X and y
        X_train, y_train = xy_split(train)
        X_test, y_test = xy_split(test)
        return X_train, X_test, y_train, y_test, train.iloc[:, :7], test.iloc[:, :7]
    else:
        #Split the random seasons to validation 
        half = len(sample) // 2
        valid_half = sample[:half]
        test_half = sample[half:]
        
        #Split data into 3 sets
        test = df[df.Season.isin(test_half)]
        train = df[~df.Season.isin(sample)]
        valid = df[df.Season.isin(valid_half)]
        
        #Split X and Y
        X_train, y_train = xy_split(train)
        X_test, y_test = xy_split(test)
        X_valid, y_valid = xy_split(valid)
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test, train.iloc[:, :7], valid.iloc[:, :7],  test.iloc[:, :7]

def weighted_error(y_pred, y_true, ref_df):
    """
    This is a custom error function used to evaluate the accuracy of the models
    It takes the rank of the predicted vote shares, and weights errors to the 1st place ranking (MVP winner) higher
    """
    
    #Add Predicted and True Outputs into Reference df
    data = ref_df[['Player', 'Season']]
    seasons = data['Season'].nunique()
    data['y_pred'] = y_pred
    data['y_true'] = y_true
    
    # Calculate predicted and true ranks by season
    data['predicted_rank'] = data.groupby('Season')['y_pred'].rank(ascending=False)
    data['true_rank'] = data.groupby('Season')['y_true'].rank(ascending=False)

    # Calculate the absolute difference between predicted and true ranks
    data['rank_diff'] = np.abs(data['predicted_rank'] - data['true_rank'])

    # Assign weights based on true_rank
    data['weight'] = np.where(data['true_rank'] == 1, 3,
                              np.where(data['true_rank'] == 2, 2, 1))

    # Calculate the weighted error
    data['weighted_error'] = data['rank_diff'] * data['weight']
    
    # Return the sum of all weighted errors
    return round(data['weighted_error'].sum() / seasons ,0)
    
def custom_obj(y_pred, dtrain):
    """
    Custom objective function for XGBoost models
    """
    y_true = dtrain.get_label()
    true_rank = np.argsort(y_true)[::-1]
    weights = np.where(true_rank == 1, 3, np.where(true_rank == 2, 2, 1))

    # Compute the gradient and hessian
    grad = (y_pred - y_true) * weights
    hess = np.ones_like(y_true) * weights

    return grad, hess

def custom_eval(model, dtrain, ref_df):
    """
    Custom evaluation metric for XGBoost models
    """
    y_true = dtrain.get_label()
    y_pred = model.predict(dtrain, output_margin=True)
    error = weighted_error(y_pred, y_true, ref_df)
    return 'weighted_error', error
    

def min_max_scale_stats(df, stats_cols):
    """
    Applies Min-Max Scaling to counting stats
    """
    
    # Select the columns to be scaled
    stats_df = df[stats_cols]
    
    # Create the scaler object
    scaler = MinMaxScaler()
    
    # Scale the selected columns
    scaled_stats = scaler.fit_transform(stats_df)
    
    # Create a new dataframe with the scaled values
    scaled_df = pd.DataFrame(scaled_stats, columns=stats_cols)
    
    # Replace the original columns in the input dataframe with the scaled values
    df[stats_cols] = scaled_df[stats_cols].round(2)
    
    return df

def df_transform(df):
    """
    This function will transform the input dataframe of season stats into the format ready for model training
    """
    
    #Calculate Win Contribution
    df['Win_Contrib'] = round(df.MP_x / 48 * df.G / 82 * df['USG%']/100 * df.Wins,1)
    
    #Drop Unneccesary Stats
    df.drop(columns = ['Pos', 'Age','GS', 'BPM', 'VORP', 'WS', 'ORB', 'DRB', 'ORB%', 'DRB%', 'eFG%', '3PAr', 'FTr', 'Wins',
                             'PF', 'FGA', '2PA', '3PA', 'FTA', 'WS/48','STL%','BLK%', 'FG', 'FG%',], inplace = True)
    #Transform Games and Minutes into %'s of totals
    df['G'] = round(df['G']/82,2)
    df['Minutes'] = round(df['MP_x']/48,2)
    df['USG%'] = round(df['USG%']/100,2)
    df.drop(columns = ['MP_x'], inplace = True)
    
    #Identify stats to min-max scale
    stats_cols = ['3P', '3P%', '2P', '2P%', 'FT', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'Win_Contrib']
    df = min_max_scale_stats(df, stats_cols)
    df = df.loc[df.Win_Contrib > 0]
    return df.sort_values(by = "Win_Contrib", ascending = False)

def join_dataframes(mvp_df, stats_df):
    """
    Joins the MVP training data to the season stats data
    """
    
    mvp_df = mvp_df.iloc[:, :8]
    df = mvp_df.merge(stats_df, on = ['Player','Season', 'Tm'])
    df.drop(columns = ['Pts Won', 'Pts Max'], inplace = True)
    df.dropna(inplace = True)
    df['Actual_Rank'] = df.groupby('Season')['Share'].rank(ascending=False, method='dense')
    last_col = df.columns[-1]  # Get the name of the last column
    col_to_move = df.pop(last_col)  # Remove the last column and save it to a variable
    df.insert(4, last_col, col_to_move)
    return df