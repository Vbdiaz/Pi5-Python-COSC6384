from multiprocessing import Event, Process
import os
from flask import Flask, request, jsonify
import tempfile
import shutil
import pandas as pd

from backgorundtask import periodic_var_calculation

from .db import get_connection
from data.montecarlo import expected_return, get_current_prices, get_portfolio_data, monte_carlo, historical_var, random_z_score, scenario_gain_loss, standard_deviation 
from datetime import datetime
from flask_cors import CORS
import yfinance as yf

app = Flask(__name__)

# CORS configuration to allow all origins and specific methods
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

@app.route('/health', methods=['GET'])
def health_check():
    """Service health check"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

## Receive trades

@app.route('/update', methods=['POST', 'OPTIONS'])
def update_transaction():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    # Main POST handler
    print("MAKE A TRAAAAAADE")
    ticker = request.json.get('ticker')
    shares = request.json.get('shares')
    action = request.json.get('action', 'buy').upper()

    if not ticker or not shares or not action:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
         # Stop background process
        print("⏹️ Stopping background process...")
        stop_event = app.config["stop_event"]
        var_process = app.config["var_process"]

        stop_event.set()
        var_process.join()  # Wait for process to fully exit

        # ✅ Get stock info from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = float(info.get("regularMarketPrice", 0))
        company = info.get("shortName", "Unknown Company")
        total_transaction_amount = round(float(shares) * current_price, 2)

        # ✅ Establish a DB connection
        db = get_connection()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500

        cursor = db.cursor()
        sql = """
            INSERT INTO transactions (
                company, ticker, transaction_type, shares,
                current_price, total_transaction_amount, transaction_date
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            company,
            ticker.upper(),
            action,
            float(shares),
            current_price,
            total_transaction_amount,
            datetime.now()
        )

        cursor.execute(sql, values)
        db.commit()
        cursor.close()
        db.close()

        # Optional: Call this only if needed
        get_portfolio_data("historical", 3, "1d")

         # Restart the background process
        print("▶️ Restarting background process...")
        new_stop_event = Event()
        new_var_process = Process(target=periodic_var_calculation, args=(new_stop_event,))
        new_var_process.start()

        app.config["var_process"] = new_var_process
        app.config["stop_event"] = new_stop_event

        return jsonify({
            'message': 'Transaction recorded',
            'company': company,
            'ticker': ticker.upper(),
            'shares': shares,
            'price': current_price,
            'total': total_transaction_amount
        })
    
        

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/var/historical', methods=['POST'])
def analyze_historical():
    """
    Receive a trade proposal and calculate hypothetical VaR
    Example POST body:
    {
        "ticker": "AAPL",
        "shares": 100,
        "action": "buy"  # or "sell"
    }
    """
    try:
        data = request.json
        ticker = data['ticker'].upper()
        shares = int(data['shares'])
        action = data['action'].lower()

        # Validate input
        if action not in ['buy', 'sell']:
            return jsonify({"error": "Invalid action. Use 'buy' or 'sell'"}), 400

        # Create temporary portfolio copy
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            # Copy original portfolio
            original_portfolio = pd.read_csv('ticker_shares.csv')
            modified_portfolio = original_portfolio.copy()

            # Find existing position
            existing = modified_portfolio[modified_portfolio['ticker'] == ticker]

            if not existing.empty:
                # Modify existing position
                idx = existing.index[0]
                if action == 'buy':
                    modified_portfolio.at[idx, 'shares'] += shares
                else:
                    if modified_portfolio.at[idx, 'shares'] < shares:
                        return jsonify({"error": "Insufficient shares to sell"}), 400
                    modified_portfolio.at[idx, 'shares'] -= shares
            else:
                # Add new position for buy actions
                if action == 'buy':
                    modified_portfolio = pd.concat([
                        modified_portfolio,
                        pd.DataFrame([[ticker, shares]], columns=['ticker', 'shares'])
                    ])
                    get_portfolio_data_api("new", 3, "1d", modified_portfolio)
                    # Calculate hypothetical VaR with temporary portfolio
                    VaR, warning, threshold, market_value = monte_carlo_api(True, modified_portfolio)

                    results = {
                        "monte_carlo": {
                            "VaR": VaR,
                            "warning": warning,
                            "threshold": threshold,
                            "market_value": market_value
                        }
                    }


                    return jsonify(results)
                else:
                    return jsonify({"error": "Cannot sell non-existent position"}), 400

            # Save temporary portfolio
            modified_portfolio.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name

        # Calculate hypothetical VaR with temporary portfolio
        VaR, warning, threshold, market_value = monte_carlo_api(False, modified_portfolio)

        results = {
            "monte_carlo": {
                "VaR": VaR,
                "warning": warning,
                "threshold": threshold,
                "market_value": market_value
            }
        }


        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'temp_path' in locals():
            os.remove(temp_path)

@app.route('/var/montecarlo', methods=['POST'])
def analyze_montecarlo():
    """
    Receive a trade proposal and calculate hypothetical VaR
    Example POST body:
    {
        "ticker": "AAPL",
        "shares": 100,
        "action": "buy"  # or "sell"
    }
    """
    try:
        data = request.json
        ticker = data['ticker'].upper()
        shares = int(data['shares'])
        action = data['action'].lower()

        # Validate input
        if action not in ['buy', 'sell']:
            return jsonify({"error": "Invalid action. Use 'buy' or 'sell'"}), 400

        # Create temporary portfolio copy
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            # Copy original portfolio
            original_portfolio = pd.read_csv('ticker_shares.csv')
            modified_portfolio = original_portfolio.copy()

            # Find existing position
            existing = modified_portfolio[modified_portfolio['ticker'] == ticker]

            if not existing.empty:
                # Modify existing position
                idx = existing.index[0]
                if action == 'buy':
                    modified_portfolio.at[idx, 'shares'] += shares
                else:
                    if modified_portfolio.at[idx, 'shares'] < shares:
                        return jsonify({"error": "Insufficient shares to sell"}), 400
                    modified_portfolio.at[idx, 'shares'] -= shares
            else:
                # Add new position for buy actions
                if action == 'buy':
                    modified_portfolio = pd.concat([
                        modified_portfolio,
                        pd.DataFrame([[ticker, shares]], columns=['ticker', 'shares'])
                    ])
                    get_portfolio_data_api("new", 3, "1d", modified_portfolio)
                    # Calculate hypothetical VaR with temporary portfolio
                    VaR, warning, threshold, market_value = monte_carlo_api(True, modified_portfolio)

                    results = {
                        "monte_carlo": {
                            "VaR": VaR,
                            "warning": warning,
                            "threshold": threshold,
                            "market_value": market_value
                        }
                    }

                    return jsonify(results)
                else:
                    return jsonify({"error": "Cannot sell non-existent position"}), 400

            # Save temporary portfolio
            modified_portfolio.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name

        # Calculate hypothetical VaR with temporary portfolio
        VaR, warning, threshold, market_value = monte_carlo_api(False, modified_portfolio)

        results = {
            "monte_carlo": {
                "VaR": VaR,
                "warning": warning,
                "threshold": threshold,
                "market_value": market_value
            }
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'temp_path' in locals():
            os.remove(temp_path)

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf


def monte_carlo_api(new_ticker: bool, ticker_shares_df):
    """Run a Monte Carlo simulation for portfolio risk analysis."""

    start_time = dt.datetime.now()

    start_data = dt.datetime.now()
    
    # Load historical price data
    if new_ticker:
        adj_close_df = pd.read_csv('new_1d_3years_data.csv', index_col=0)
    else:
        adj_close_df = pd.read_csv('historical_1d_3years_data.csv', index_col=0)
    tickers = list(adj_close_df.columns)

    print(f"Tickers: {tickers}")

    # Get current prices and append as last row
    current_prices = get_current_prices(tickers)
    current_prices_df = pd.DataFrame([current_prices], index=[dt.datetime.now()])
    adj_close_df = pd.concat([adj_close_df, current_prices_df])

    end_data = dt.datetime.now()
    duration = end_data - start_data
    print(f"start: {start_data} end: {end_data} duration: {duration}")

    # Compute daily log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # Load share holdings

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
    simulations = 100000
    days = 20
    scenario_returns = [
        scenario_gain_loss(market_value, portfolio_expected_return, portfolio_std_dev, random_z_score(), days)
        for _ in range(simulations)
    ]

    # Compute Value at Risk (VaR) at 95% confidence interval
    confidence_interval = 0.99
    VaR = -np.percentile(scenario_returns, 100 * (1 - confidence_interval))

    # Expected Shortfall (ES): mean of losses worse than the VaR
    losses = [x for x in scenario_returns if x < -VaR]
    ES = -np.mean(losses) if losses else 0
    end_time = dt.datetime.now()

    print(f"MC (ES) at {confidence_interval * 100}% confidence: ${ES:,.2f}")
    print(f"MC (VaR) at {confidence_interval * 100}% confidence: ${VaR:,.2f}")

    threshold = 0.11
    percent_at_risk = VaR / market_value
    warning = True if percent_at_risk >= threshold else False

    return VaR, warning, threshold, market_value

def get_portfolio_data_api(name: str, years: int, interval: str, shares_csv):
    """
    Fetches portfolio data from MySQL, saves it to CSV, and downloads historical stock prices.

    Args:
        years (int): Number of years of historical data to fetch.
    """

    # Extract tickers
    tickers = shares_csv["ticker"].tolist()

    # Define date range
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * years)

    adj_close_df = pd.DataFrame()

    # Download historical data for each ticker
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if not stock_data.empty:
            adj_close_df[ticker] = stock_data["Close"]

    # Save historical data to CSV
    adj_close_df.to_csv(f"{name}_{interval}_{years}years_data.csv", index=True)


def historical_var_api(new_ticker: bool, ticker_shares_df):
    """Calculate Value at Risk using Historical Simulation method"""

    start_time = dt.datetime.now()

    # Load historical price data
    if new_ticker:
        adj_close_df = pd.read_csv('new_1d_3years_data.csv', index_col=0)
    else:
        adj_close_df = pd.read_csv('historical_1d_3years_data.csv', index_col=0)
    tickers = list(adj_close_df.columns)
    
    # Get current prices and update DataFrame
    current_prices = get_current_prices(tickers)
    current_prices_df = pd.DataFrame([current_prices], index=[dt.datetime.now()])
    adj_close_df = pd.concat([adj_close_df, current_prices_df])

    # Calculate daily returns
    returns = adj_close_df.pct_change().dropna()

    # Load portfolio weights
    shares = ticker_shares_df['shares'].values
    prices = adj_close_df.iloc[-1].values
    portfolio_value = np.sum(shares * prices)
    weights = (shares * prices) / portfolio_value

    # Calculate historical portfolio returns
    portfolio_returns = returns.dot(weights)

    days = 20

    # Calculate 20-day rolling returns
    portfolio_returns_window = portfolio_returns.rolling(window=days).sum().dropna()

    # Generate hypothetical P&L scenarios
    scenarios = portfolio_value * portfolio_returns_window

    # Calculate VaR
    confidence_interval = 0.99
    VaR = -np.percentile(scenarios, 100*(1-confidence_interval))

    # Expected Shortfall (ES)
    losses = scenarios[scenarios < -VaR]
    ES = -np.mean(losses) if not losses.empty else 0
    end_time = dt.datetime.now()

    print(f"Historical (ES) at {confidence_interval * 100}% confidence: ${ES:,.2f}")
    print(f"Historical (VaR) at {confidence_interval * 100}% confidence: ${VaR:,.2f}")

    threshold = 0.095
    percent_at_risk = VaR/portfolio_value
    warning = True if percent_at_risk >= threshold else False
    
    return VaR, warning, threshold, portfolio_value