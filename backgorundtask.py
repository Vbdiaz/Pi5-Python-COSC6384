from datetime import datetime, timedelta
import time

from data.montecarlo import historical_var, monte_carlo


MARKET_OPEN = datetime.strptime("00:00", "%H:%M").time()
MARKET_CLOSE = datetime.strptime("23:59", "%H:%M").time()

def get_next_open_time():
    now = datetime.now()
    next_open = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    if now.time() > MARKET_OPEN:
        next_open += datetime.timedelta(days=1)
    return next_open

def is_market_open():
    now = datetime.now().time()
    return MARKET_OPEN <= now < MARKET_CLOSE

def periodic_var_calculation(stop_event):
    while not stop_event.is_set():
        if is_market_open():
            start = time.time()
            print(f"MARKET OPEN: Calculating VaR Time: {start}")

            # If we need to stop, break out before heavy processing
            if stop_event.is_set(): break

            start_time = time.time()
            monte_carlo()
            print("MonteCarlo: --- %s seconds ---" % (time.time() - start_time))

            if stop_event.is_set(): break

            start_time = time.time()
            historical_var()
            print("Historical: --- %s seconds ---" % (time.time() - start_time))

            time_to_next_minute = time.time() - start
            time.sleep(max(0, 59 - time_to_next_minute))
        else:
            sleep_time = (get_next_open_time() - datetime.now()).total_seconds()
            print(f"MARKET CLOSED: Sleeping until next open time for {sleep_time} seconds...")
            time.sleep(sleep_time)
