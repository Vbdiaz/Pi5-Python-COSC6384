from multiprocessing import Process, Event
from api import create_app

from backgorundtask import periodic_var_calculation
from data.montecarlo import get_portfolio_data

app = create_app()

# Global references
var_process = None
stop_event = None

if __name__ == "__main__":
    stop_event = Event()
    get_portfolio_data("historical", 3, "1d")

    var_process = Process(target=periodic_var_calculation, args=(stop_event,))
    var_process.start()

    app.config["var_process"] = var_process
    app.config["stop_event"] = stop_event

    app.run(host='0.0.0.0', port=5001)
