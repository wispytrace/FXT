from .base import *
import numpy as np


class Show():
    
    def __init__(self) -> None:
        self.folder = 'output/signal_recover'
        self.compare_folder = self.folder + '/compare/'
        self.file_folder = self.folder + '/{}/'
        self.file = self.folder + '/{}/result.txt'
        os.makedirs(self.compare_folder, exist_ok=True)
        os.makedirs(self.file_folder, exist_ok=True)

    
    def extract_key_value(self, key, file_path):
        memory = extract_records(file_path)
        algorithm = memory['global']['algorithm']
        time = memory['data']['time']
        value = memory['data'][key]

        return time, value, algorithm
    
    def show_compare(self, indexs):
        times = []
        values = []
        algorithms = []
        file_name = ''
        file = open(self.compare_folder+"convergence_time.txt", mode='w', encoding='utf-8')
        for index in indexs:
            time, value, algorithm = self.extract_key_value('diffvalue', self.file.format(index))
            self.save_convergence_time(file, index, value, time)
            value = np.log10(value)
            values.append(value)
            algorithms.append(algorithm)
            file_name += '_'+str(index)
            times.append(time)
        # plot_graph(times, values, '$lg||x-x^{\star}||$', 'time(sec)', labels=["DS", "FiNN", "FxNN", "NFxNN"],x_lim=[0, 0.5])

        plot_graph(times, values, '$lg||x-x^{\star}||$', 'time(sec)', labels=["DS", "FiNN", "FxNN", "NFxNN", "TFxND"],x_lim=[0, 0.5])
        plt.savefig(self.compare_folder+file_name+'.jpeg')
        plt.savefig(self.compare_folder+file_name+'.eps')
        file.flush()
        file.close()
                
    
    def save_convergence_time(self, file, index, value, time, error_bound=1e-2):
        convergence_time = -1
        for i in range(len(value)-1):
            relative_error = (value[i+1] - value[-1])/value[-1]
            if np.fabs(relative_error) < error_bound:
                convergence_time = time[i+1]
                break

        file.write("{} settle_time: {} \n".format(index, convergence_time))
        
    
    def show_error(self, index):
        file_path = self.file.format(index)
        time, value, _ = self.extract_key_value('diffvalue',file_path)
        plot_graph([time], [value], '$||z-x||$', 'time(sec)', None)
        plt.savefig(self.file_folder.format(index)+'error.jpeg')
        
    def show_wave(self, index):
        file_path = self.file.format(index)
        memory = extract_records(file_path)
        wave_true = memory['global']['true_value']
        wave_observed = memory['global']['observed_value']
        plot_graph([[i for i in range(len(wave_true))]] , [wave_true], None, None, None, x_lim=[0, 512])
        plt.yticks([])  # 去掉y轴的刻度标签
        plt.xticks([])  
        plt.savefig(self.file_folder.format(index)+'wave_true.jpeg')
        plt.savefig(self.file_folder.format(index)+'wave_true.eps')
        plot_graph([[i for i in range(len(wave_observed))]] , [wave_observed], None, None, None, x_lim=[0, 256])
        plt.yticks([])  # 去掉y轴的刻度标签
        plt.xticks([])  
        plt.savefig(self.file_folder.format(index)+'wave_observed.jpeg')
        plt.savefig(self.file_folder.format(index)+'wave_observed.eps')
        wave_recover = memory['data']['state'][-200]
        plot_graph([[i for i in range(len(wave_recover))]] , [wave_recover], None, None, None, x_lim=[0, 512])
        plt.yticks([])  # 去掉y轴的刻度标签 
        plt.xticks([])  
        plt.savefig(self.file_folder.format(index)+'wave_recover.jpeg')
        plt.savefig(self.file_folder.format(index)+'wave_recover.eps')

    def show(self, index):
        # self.show_error(index)
        # self.show_wave(index)
        # self.show_compare(['4_1', '4_2', '4_3', '4_4', '4'])
        # self.show_compare(['5_1', '5_2', '5_3', '5_4', '5'])
        self.show_compare(['6', '6_1', '6_2', '6_3', '6_4'])
        # self.show_compare(['5_1', '5_2', '5_3', '5_4', '5'])
