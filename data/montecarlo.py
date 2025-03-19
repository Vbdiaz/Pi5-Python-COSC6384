#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from scipy.stats import norm

from api.db import get_connection

def getHistoricalData(years):
    mydb = get_connection()

    if mydb:
        mycursor = mydb.cursor()

        mycursor.execute("SELECT ticker FROM portfolio")

        tickers = [ticker[0] for ticker in mycursor.fetchall()]

        print(type(tickers))
        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=365 * years)

        adj_close_df = pd.DataFrame()

        # Download historical data excluding the current day
        for ticker in tickers:
            data = yf.download(ticker, start=startDate, end=endDate)
            adj_close_df[ticker] = data['Close']
        
        adj_close_df.to_csv('historical_data.csv')

    else:
        print("Database connection failed. montecarlo.py")

def montecarlo():
    adj_close_df = pd.read_csv('historical_data.csv', index_col=0, parse_dates=True)
    tickers = list(adj_close_df.columns)[1:]

    print(tickers)

    # Get current prices
    current_prices = {}
    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)
        current_price = ticker_data.history(period='1d').tail(1)['Close'].values[0]
        current_prices[ticker] = current_price

    # Append the current prices as a new row
    current_prices_df = pd.DataFrame([current_prices], index=[dt.datetime.now()])
    adj_close_df = pd.concat([adj_close_df, current_prices_df])

    print(adj_close_df.tail(1))

    ### Calculate the daily log returns and drop any NAs
    log_returns = np.log(adj_close_df/adj_close_df.shift(1))
    log_returns = log_returns.dropna()

    # print(log_returns)

    ### Create a function that will be used to calculate portfolio expected return
    ### We are assuming that future returns are based on past returns, which is not a reliable assumption.
    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean()*weights)

    ### Create a function that will be used to calculate portfolio standard deviation
    def standard_deviation (weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)


    ### Create a covariance matrix for all the securities
    cov_matrix = log_returns.cov()
    # print(cov_matrix)


    ### Create an equally weighted portfolio and find total portfolio expected return and standard deviation
    portfolio_value = 1000000
    weights = np.array([
    0.085, 0.054, 0.029, 0.032, 0.034, 0.054, 0.037, 0.041, 
    0.032, 0.016, 0.062, 0.053, 0.037, 0.040, 0.009, 0.047, 
    0.053, 0.020, 0.046, 0.035, 0.039, 0.068, 0.030, 0.054, 
    0.031, 0.085, 0.044, 0.016, 0.013, 0.048])
    portfolio_expected_return = expected_return(weights, log_returns)
    portfolio_std_dev = standard_deviation(weights, cov_matrix)


    def random_z_score():
        return np.random.normal(0,1)

    ### Create a function to calculate scenarioGainLoss
    days = 20

    def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
        return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score * np.sqrt(days)

    ### Run 10000 simulations
    simulations = 20000
    scenarioReturn = []

    for i in range (simulations):
        z_score = random_z_score()
        scenarioReturn.append(scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days))

    ### Specify a confidence interval and calculate the Value at Risk (VaR)
    confidence_interval = .95
    VaR = -np.percentile(scenarioReturn, 100 * (1 - confidence_interval))
    print(VaR)