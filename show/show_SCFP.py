from .base import *
import numpy as np


class Show():

    def __init__(self) -> None:
        self.folder = 'output/SCFP'
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
        # value = np.log(memory['data'][key])

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
        plot_graph(times, values, '$log_10||x-x^{\star}||$', 'time(sec)', labels=["DS", "FiNN", "FxNN", "TFxND"])
        plt.savefig(self.compare_folder+file_name+'.jpeg')
        plt.savefig(self.compare_folder+file_name+'.eps')
                
    
    def show_error(self, index):
        file_path = self.file.format(index)
        time, value, _ = self.extract_key_value('diffvalue',file_path)
        plot_graph([time], [value], '$||z-x||$', 'time(sec)', None)
        plt.savefig(self.file_folder.format(index)+'error.jpeg')
        
    def show_value(self, index):
        file_path = self.file.format(index)
        time, x, _ = self.extract_key_value('state',file_path)
        labels = ["$x_1$", "$x_2$", "$x_3$"]
        plot_graph([time, time, time], [x[:,0],x[:,1], x[:,2]], '$x(t)$', 'time(sec)', labels=labels, x_lim=[0, 0.25])
        plt.savefig(self.file_folder.format(index)+'value.eps')


    def show(self, index):
        self.show_value(index)
        self.show_compare(['2_1', '2_2', '2_3', '2'])
