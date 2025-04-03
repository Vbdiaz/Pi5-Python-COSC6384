import os
from flask import Flask, request, jsonify
import tempfile
import shutil
import pandas as pd
from api.db import get_connection
from data.montecarlo import monte_carlo, historical_var 
from datetime import datetime


app = Flask(__name__)

@app.route('/var/latest', methods=['GET'])
def get_latest_var():
    """Get the latest VaR calculation from database"""
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT * FROM value_at_risk 
            ORDER BY calculation_time DESC 
            LIMIT 1
        """)
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            result['calculation_time'] = result['calculation_time'].isoformat()
            return jsonify(result)
        return jsonify({"message": "No VaR records found"}), 404
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Service health check"""
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

## Receive trades

@app.route('/var/analyze-trade', methods=['POST'])
def analyze_trade():
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
                else:
                    return jsonify({"error": "Cannot sell non-existent position"}), 400

            # Save temporary portfolio
            modified_portfolio.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name

        # Calculate hypothetical VaR with temporary portfolio
        results = {
            "original_var": get_current_var(),
            "hypothetical": {
                "monte_carlo": calculate_hypothetical_var(temp_path, monte_carlo),
                "historical": calculate_hypothetical_var(temp_path, historical_var)
            }
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'temp_path' in locals():
            os.remove(temp_path)

def calculate_hypothetical_var(temp_portfolio_path, var_function):
    """Calculate VaR with modified portfolio"""
    try:
        # Backup original files
        shutil.copy('ticker_shares.csv', 'ticker_shares.csv.bak')
        shutil.copy(temp_portfolio_path, 'ticker_shares.csv')
        
        return var_function()
    finally:
        # Restore original portfolio
        shutil.move('ticker_shares.csv.bak', 'ticker_shares.csv')

def get_current_var():
    """Get latest VaR from database"""
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM value_at_risk 
            ORDER BY calculation_time DESC 
            LIMIT 1
        """)
        return cursor.fetchone()
    except Exception as e:
        print(f"Error getting current VaR: {e}")
        return None