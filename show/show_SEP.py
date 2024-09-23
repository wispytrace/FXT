from .base import *
import numpy as np
import os

class Show():
    
    def __init__(self) -> None:
        self.folder = 'output/SEP'
        self.compare_folder = self.folder + '/compare/'
        self.file_folder = self.folder + '/{}/'
        self.file = self.folder + '/{}/result.txt'
        os.makedirs(self.compare_folder, exist_ok=True)
        os.makedirs(self.file_folder, exist_ok=True)

    
    def extract_key_value(self, key, file_path):
        memory = extract_records(file_path)
        algorithm = memory['global']['algorithm']
        time = memory['data']['time']
        value = np.array(memory['data'][key])

        return time, value, algorithm
    
    def show_compare(self, indexs):
        times = []
        values = []
        algorithms = []
        file_name = ''
        for index in indexs:
            time, value, algorithm = self.extract_key_value('diffvalue', self.file.format(index))
            value = np.log10(value)
            values.append(value)
            algorithms.append(algorithm)
            file_name += '_'+str(index)
            times.append(time)
        plot_graph(times, values, '$log_10||z-z^{\star}||$', 'time(sec)', labels=["TFxND", "DS"])
        plt.savefig(self.compare_folder+file_name+'.eps')
        plt.savefig(self.compare_folder+file_name+'.jpeg')
                
    
    def show_error(self, index):
        file_path = self.file.format(index)
        time, value, _ = self.extract_key_value('diffvalue',file_path)
        plot_graph([time], [value], '$z-x$', 'time(sec)', None)
        plt.savefig(self.file_folder.format(index)+'error.eps')
        
    def show_value(self, index):
        file_path = self.file.format(index)
        time, x, _ = self.extract_key_value('x',file_path)
        time, y, _ = self.extract_key_value('y',file_path)
        labels = ["$z_1$", "$z_2$", "$z_3$", "$z_4$"]
        plot_graph([time, time, time, time], [x[:,0],x[:,1],y[:,0],y[:,1]], '$z(t)$', 'time(sec)', labels=labels, y_lim=[-1, 25])
        plt.savefig(self.file_folder.format(index)+'value.eps')
        plt.savefig(self.file_folder.format(index)+'value.jpeg')

    def show(self, index):
        self.show_value(index)
        # self.show_error(index)
        self.show_compare(['2','2_3'])

        
