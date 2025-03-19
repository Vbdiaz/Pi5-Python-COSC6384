import multiprocessing
from api import create_app
import threading
import time
from datetime import datetime, timedelta

from data.montecarlo import get_portfolio_data, monte_carlo

MARKET_OPEN = datetime.strptime("08:30", "%H:%M").time()
MARKET_CLOSE = datetime.strptime("15:00", "%H:%M").time()

def get_next_open_time():
    now = datetime.now()
    next_open = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    if now.time() > MARKET_OPEN:
        next_open += timedelta(days=1)
    return next_open

def is_market_open():
    now = datetime.now().time()
    return MARKET_OPEN <= now < MARKET_CLOSE

def periodic_var_calculation():
    while True:
        print("Starting periodic VaR calculation...")
        if is_market_open():
            print("MARKET OPEN: Calculating VaR, sleeping for 10 seconds...")
            monte_carlo()
            time.sleep(10)
        else:
            # Save last three years of data after market closes
            sleep_time = (get_next_open_time() - datetime.now()).total_seconds()
            print(f"MARKET CLOSED: Sleeping until next open time for {sleep_time} seconds...")
            time.sleep(sleep_time)
            

app = create_app()

if __name__ == "__main__":
    # Save last three years of data everytime we start again
    get_portfolio_data(3)
    var_process = multiprocessing.Process(target=periodic_var_calculation)
    var_process.start()
    #periodic_var_calculation()
    app.run(host='0.0.0.0', port=5000)
