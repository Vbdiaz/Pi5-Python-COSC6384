import multiprocessing
from api import create_app
import time
from datetime import datetime, timedelta

from data.montecarlo import get_portfolio_data, monte_carlo, historical_var

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
        if is_market_open():
            start = time.time()
            print(f"MARKET OPEN: Calculating VaR Time: {start}")
            start_time = time.time()
            monte_carlo()
            print("MonteCarlo: --- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            historical_var()
            print("Historical: --- %s seconds ---" % (time.time() - start_time))
            time_to_next_minute = time.time() - start
            time.sleep(max(0, 59 - time_to_next_minute))
        else:
            # Save last three years of data after market closes
            sleep_time = (get_next_open_time() - datetime.now()).total_seconds()
            time.sleep(sleep_time)
            print(f"MARKET CLOSED: Sleeping until next open time for {sleep_time} seconds...")
            

app = create_app()

if __name__ == "__main__":
    # Save last three years of data everytime we start again
    get_portfolio_data("historical", 3, "1d")
    var_process = multiprocessing.Process(target=periodic_var_calculation)
    var_process.start()
    #periodic_var_calculation()
    app.run(host='0.0.0.0', port=5001)
