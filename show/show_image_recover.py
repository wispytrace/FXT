from .base import *
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import copy

class Show():
    
    def __init__(self) -> None:
        self.folder = 'output/image_recover'
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
    
    def extract_psnr_ssim(self, file_path):
        memory = extract_records(file_path)
        algorithm  =  memory['global']['algorithm']
        image = memory['global']['image']
        image = np.array(image).astype(np.uint8)
        time = memory['data']['time']
        state = memory['data']['state']
        psnr_value = []
        ssim_value = []
        for i in range(len(state)):
            state_image = np.array(state[i]).reshape(128, 128).astype(np.uint8)
            psnr_value.append(peak_signal_noise_ratio(image, state_image))
            ssim_value.append(structural_similarity(image, state_image))
        
        return time, psnr_value, ssim_value, algorithm
        
    
    def show_compare(self, indexs):
        times = []
        values = []
        algorithms = []
        file_name = ''
        for index in indexs:
            time, value, algorithm = self.extract_key_value('diffvalue', self.file.format(index))
            values.append(np.log(value))
            algorithms.append(algorithm)
            file_name += '_'+str(index)
            times.append(time)
        plot_graph(times, values, '$log_2||x-x^{\star}||$', 'time(sec)', labels=["DS", "FiNN", "FxNN", "TFxND"])
        plt.savefig(self.compare_folder+file_name+'.jpeg')
        plt.savefig(self.compare_folder+file_name+'.eps')
    
    
    def show_psnr_ssim_compare(self, indexs):
        times = []
        psnr_values = []
        ssim_values = []
        algorithms = []
        file_name = ''
        for index in indexs:
            time, psnr_value, ssim_value, algorithm = self.extract_psnr_ssim(self.file.format(index))
            psnr_values.append(copy.deepcopy(psnr_value))
            ssim_values.append(copy.deepcopy(ssim_value))
            algorithms.append(algorithm)
            file_name += '_'+str(index)
            times.append(time)

        plot_graph(times, psnr_values, 'PSNR', 'time(sec)', labels=["DS", "FiNN", "FxNN", "NFXNN", "TFxND"], x_lim=[0, 3])
        plt.savefig(self.compare_folder+file_name+'psnr.jpeg')
        plt.savefig(self.compare_folder+file_name+'psnr.eps')
        plot_graph(times, ssim_values, 'SSIM', 'time(sec)', labels=["DS", "FiNN", "FxNN", "NFXNN", "TFxND"], y_lim=[0, 1], x_lim=[0,3])
        plt.savefig(self.compare_folder+file_name+'ssim.jpeg')
        plt.savefig(self.compare_folder+file_name+'ssim.eps')
        
    
    
    def show_error(self, index):
        file_path = self.file.format(index)
        time, value, _ = self.extract_key_value('diffvalue',file_path)
        plot_graph([time], [value], '$||z-x||$', 'time(sec)', None)
        plt.savefig(self.file_folder.format(index)+'error.jpeg')

    def show(self, index):
        # self.show_error(index)
        self.show_psnr_ssim_compare(['2_1', '2_2', '2_4', '2_3', '2'])
        # self.show_compare(['2_1', '2_2', '2_4', '2_3', '2'])
