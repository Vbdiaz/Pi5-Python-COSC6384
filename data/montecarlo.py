import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from api.db import get_connection

def get_portfolio_data(years: int):
    """
    Fetches portfolio data from MySQL, saves it to CSV, and downloads historical stock prices.

    Args:
        years (int): Number of years of historical data to fetch.
    """
    try:
        # Establish database connection
        mydb = get_connection()
        with mydb.cursor() as mycursor:
            # Execute query
            query = "SELECT ticker, shares FROM portfolio ORDER BY ticker ASC"
            mycursor.execute(query)

            # Fetch results
            rows = mycursor.fetchall()

            # Convert to DataFrame
            ticker_shares_df = pd.DataFrame(rows, columns=["ticker", "shares"])

            # Save portfolio data to CSV
            ticker_shares_df.to_csv("ticker_shares.csv", index=False)

        # Extract tickers
        tickers = ticker_shares_df["ticker"].tolist()

        # Define date range
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365 * years)

        adj_close_df = pd.DataFrame()

        # Download historical data for each ticker
        for ticker in tickers:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                adj_close_df[ticker] = stock_data["Close"]

        # Save historical data to CSV
        adj_close_df.to_csv("historical_data.csv", index=True)

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if mydb.is_connected():
            mydb.close()

def get_current_prices(tickers):
    """Fetch the latest closing prices for the given tickers."""
    current_prices = {}
    for ticker in tickers:
        try:
            ticker_data = yf.Ticker(ticker)
            current_price = ticker_data.history(period='1d').tail(1)['Close'].values[0]
            current_prices[ticker] = current_price
        except Exception as e:
            print(f"Error retrieving price for {ticker}: {e}")
            current_prices[ticker] = np.nan  # Handle missing data

    return current_prices


def expected_return(weights, log_returns):
    """Calculate the expected portfolio return."""
    return np.sum(log_returns.mean() * weights)


def standard_deviation(weights, cov_matrix):
    """Calculate the portfolio standard deviation."""
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)


def random_z_score():
    """Generate a random Z-score from a standard normal distribution."""
    return np.random.normal(0, 1)


def scenario_gain_loss(market_value, portfolio_expected_return, portfolio_std_dev, z_score, days):
    """Calculate portfolio gain/loss over a given period."""
    return (
        market_value * portfolio_expected_return * days +
        market_value * portfolio_std_dev * z_score * np.sqrt(days)
    )

def monte_carlo():
    """Run a Monte Carlo simulation for portfolio risk analysis."""
    
    # Load historical price data
    adj_close_df = pd.read_csv('historical_data.csv', index_col=0)
    tickers = list(adj_close_df.columns)

    print(f"Tickers: {tickers}")

    # Get current prices and append as last row
    current_prices = get_current_prices(tickers)
    current_prices_df = pd.DataFrame([current_prices], index=[dt.datetime.now()])
    adj_close_df = pd.concat([adj_close_df, current_prices_df])

    # Compute daily log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # Load share holdings
    ticker_shares_df = pd.read_csv('ticker_shares.csv')

    # Ensure tickers are in order
    if ticker_shares_df['ticker'].tolist() != tickers:
        print("Error: Tickers in 'ticker_shares.csv' are out of order.")
        return

    # Compute portfolio value and weights
    shares_held = ticker_shares_df.iloc[:, 1].values  # Get shares column
    prices_now = adj_close_df.iloc[-1, :].values  # Get latest prices
    market_value = np.sum(shares_held * prices_now)

    if market_value == 0:
        print("Error: Portfolio value is zero, check holdings or price data.")
        return

    weights = (shares_held * prices_now) / market_value

    print(f"Weights: {weights}")
    print(f"Market Value: {market_value}")

    # Compute portfolio metrics
    cov_matrix = log_returns.cov()
    portfolio_expected_return = expected_return(weights, log_returns)
    portfolio_std_dev = standard_deviation(weights, cov_matrix)

    # Monte Carlo simulation
    simulations = 20000
    days = 20
    scenario_returns = [
        scenario_gain_loss(market_value, portfolio_expected_return, portfolio_std_dev, random_z_score(), days)
        for _ in range(simulations)
    ]

    # Compute Value at Risk (VaR) at 95% confidence interval
    confidence_interval = 0.95
    VaR = -np.percentile(scenario_returns, 100 * (1 - confidence_interval))

    print(f"Value at Risk (VaR) at {confidence_interval * 100}% confidence: ${VaR:,.2f}")

    threshold = 0.0677
    percent_at_risk = VaR / market_value
    warning = True if percent_at_risk >= threshold else False

    push_value_at_risk_data(VaR, "monte_carlo", tickers, prices_now, market_value, warning, percent_at_risk, threshold)
    return VaR

def push_value_at_risk_data(VaR: float, method: str, tickers: list, prices: list, portfolio_value: float, warning, percent_at_risk: float, threshold: float):
    """
    Push portfolio VaR data and corresponding stock data into the database.

    Args:
        VaR (float): Calculated portfolio Value at Risk.
        method (str): The method used for calculating VaR (e.g., "monte_carlo").
        tickers (list): List of stock tickers.
        prices (list): List of current stock prices for each ticker.
        portfolio_value (float): The total portfolio market value.
    """
    try:
        # Establish database connection
        mydb = get_connection()
        with mydb.cursor() as mycursor:
            # Get the current timestamp; this will be our unique identifier
            current_time = dt.datetime.now()

            # Convert VaR to a native Python float in case it's a NumPy type.
            VaR_native = float(VaR)
            portfolio_value_native = float(portfolio_value)
            percent_at_risk_native = float(percent_at_risk)
            threshold_native = float(threshold)
            
            # Insert the portfolio-wide VaR record
            sql_value_at_risk = """
                INSERT INTO value_at_risk (calculation_time, var_value, method, portfolio_value, warning, percent_at_risk, percent_threshold)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            mycursor.execute(sql_value_at_risk, (current_time, VaR_native, method, portfolio_value_native, warning, percent_at_risk_native, threshold_native))

            # Prepare the SQL for inserting stock data
            sql_stock_data = """
                INSERT INTO stock_data_log (calculation_time, ticker, current_price)
                VALUES (%s, %s, %s)
            """
            # Build a list of tuples for each stock record
            stock_records = [
                (current_time, tickers[i], float(prices[i]))
                for i in range(len(tickers))
            ]
            mycursor.executemany(sql_stock_data, stock_records)

            # Commit the transaction
            mydb.commit()

            print(f"Successfully inserted VaR record and {len(tickers)} stock records with timestamp {current_time}.")

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if mydb.is_connected():
            mydb.close()

## Second VAR calculation Method

def historical_var():
    """Calculate Value at Risk using Historical Simulation method"""
    
    # Load data
    adj_close_df = pd.read_csv('historical_data.csv', index_col=0)
    tickers = list(adj_close_df.columns)
    
    # Get current prices and update DataFrame
    current_prices = get_current_prices(tickers)
    current_prices_df = pd.DataFrame([current_prices], index=[dt.datetime.now()])
    adj_close_df = pd.concat([adj_close_df, current_prices_df])

    # Calculate daily returns
    returns = adj_close_df.pct_change().dropna()

    # Load portfolio weights
    ticker_shares_df = pd.read_csv('ticker_shares.csv')
    shares = ticker_shares_df['shares'].values
    prices = adj_close_df.iloc[-1].values
    portfolio_value = np.sum(shares * prices)
    weights = (shares * prices) / portfolio_value

    # Calculate historical portfolio returns
    portfolio_returns = returns.dot(weights)

    # Generate hypothetical P&L scenarios
    scenarios = portfolio_value * portfolio_returns

    # Calculate VaR
    confidence = 0.95
    VaR = -np.percentile(scenarios, 100*(1-confidence))

    print(f"Historical VaR (95% CI): ${VaR:,.2f}")

    threshold = 0.0149
    percent_at_risk = VaR/portfolio_value
    warning = True if percent_at_risk >= threshold else False
    
    push_value_at_risk_data(
        VaR, 
        "historical", 
        tickers,
        prices,
        portfolio_value, 
        warning, 
        percent_at_risk, 
        threshold
    )
    return VaR