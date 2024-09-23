import argparse
from show import Shows
import time



def show_results(opt):
    show = Shows[opt.example]()
    show.show(opt.config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--example", type=str, default='signal_recover')
    parser.add_argument('-c', "--config", type=str, default='5_3')

    opt = parser.parse_args()
    show_results(opt)
