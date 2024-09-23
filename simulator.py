import argparse
from examples import Examples
from algorithms import Algorithms
import random
import time

PROCESS_BAR_INTERVAL = 100

def seconds_to_hms_string(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{round(seconds,2):04}"

def publish_process_bar(counts, start_time, epochs, time_estimate):
    now_time = time.time()
    interval = now_time - time_estimate
    total_time = (epochs / PROCESS_BAR_INTERVAL) * interval
    used_time = now_time - start_time
    total_time = seconds_to_hms_string(total_time)
    used_time = seconds_to_hms_string(used_time)
    square_num = int(int(counts / epochs * 100) / 10)
    square = "■" * square_num
    blank = " " * (10 -square_num)
    bar = f"|{square}{blank}|"
    ratio = round(counts / epochs * 100, 3) 
    message = f"{ratio:5}% {bar} {str(counts) + '/' + str(epochs):15} [{used_time}<{total_time}, {round(interval,2)}s/{PROCESS_BAR_INTERVAL}its]"
    # self.get_logger().info('\033[2J\033[;H')
    print(message)
    time_estimate = now_time
    return time_estimate

def run_simulation(opt):
    example = Examples[opt.example]()
    example.load_config(opt.config, opt.time_delta)
    algorithm = Algorithms[example.config['algorithm']]()
    example.apply_optimizer(algorithm)
    start_time = time.time()
    time_estimate = start_time
    for i in range(opt.epochs):
        example.update()
        if i % PROCESS_BAR_INTERVAL == 0:
            time_estimate = publish_process_bar(i, start_time, opt.epochs, time_estimate)
    example.save()

def batch_run(configs_list):
    for config in configs_list:
        random.seed(666)
        parser = argparse.ArgumentParser()
        parser.add_argument('-ex', "--example", type=str, default='signal_recover')
        parser.add_argument('-ep',"--epochs", type=int, default=5000)
        # parser.add_argument('-a',"--algorithm", type=str, default='FXT')
        parser.add_argument('-c', "--config", type=str, default=config)
        parser.add_argument('-t', "--time_delta", type=int, default=1e-4)
        opt = parser.parse_args()
        run_simulation(opt)


if __name__ == "__main__":
    # image 2000 5e-6
    # signal 600 5e-4
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-ex', "--example", type=str, default='signal_recover')
    # parser.add_argument('-ep',"--epochs", type=int, default=10000)
    # # parser.add_argument('-a',"--algorithm", type=str, default='FXT')
    # parser.add_argument('-c', "--config", type=str, default='6_2')
    # parser.add_argument('-t', "--time_delta", type=int, default=1e-4)

    # opt = parser.parse_args()
    # run_simulation(opt)
    batch_run(['6_3'])
