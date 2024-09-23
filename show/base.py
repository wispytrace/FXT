import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

plt.rcParams['figure.dpi'] = 600 
plt.rcParams['lines.linewidth'] = 0.5


def make_folder(path):
    os.makedirs(path, exist_ok=True)    

def extract_records(record_path):
    with open(record_path, 'r') as f:
        records = json.loads(f.read())
        memory = defaultdict()
        memory['global'] = records['global']
        memory['data'] = {}
        for i, record in enumerate(records['data']):
            for k, v in record.items():
                if k not in memory['data'].keys():
                    memory['data'][k] = []
                memory['data'][k].append(v)
    return memory


def plot_graph(x_list, y_list, y_title, x_title, labels, x_lim=None, y_lim=None, label_box=None, font_szie=None):
    plt.clf()
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(len(y_list)):
        if labels == None:
            plt.plot(x_list[i], y_list[i], color=mcolors.TABLEAU_COLORS[colors[i%(len(colors))]])
        else:
            plt.plot(x_list[i], y_list[i], color=mcolors.TABLEAU_COLORS[colors[i%(len(colors))]], label=labels[i])
            
    plt.xlim(0)
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    if x_title is not None:
        plt.xlabel(x_title, fontsize=15)
    if y_title is not None:
        plt.ylabel(y_title, fontsize=15)
    if font_szie is not None:
        plt.rcParams.update({'font.size':font_szie}) 
    if label_box is not None and labels is not None:
        plt.legend(loc='lower left', bbox_to_anchor=(label_box[0], label_box[1]), fontsize=20)
    if labels is not None:
        plt.legend()
