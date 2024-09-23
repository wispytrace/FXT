from .base import *
import cv2 as cv
import numpy as np
import os
from scipy.io import loadmat


faces = loadmat('resources\image_recover\olivettifaces.mat')['faces']
model_configs = {
    "0":{
        'kernel': np.array([[1, 2, 1],  [2, 4, 2],  [1, 2, 1]]) / 16,
        'image': faces[:,0].reshape(64, 64),
        'C': SetLevel([0]*64*64, [255]*64*64)
    },
    "1":{
        # 'kernel': np.array([[1, 2, 1],  [2, 4, 2],  [1, 2, 1]]) / 16,
        'kernel': Matrix.gaussian_kernel(5),
        'image': cv.cvtColor(cv.imread('resources\image_recover\original\cat.jpg'), cv.COLOR_BGR2GRAY),
        'C': SetLevel([0]*128*128, [255]*128*128)
    },
    
}

configs = {
    "0": {
        'initial_state': np.zeros((64*64)),
        'model': model_configs['0'],
        'gama': 1e-2,
        'parameters':{'c1': 1, 'c2': 1, "p": 0.5, "q":1.5},
        'algorithm': "FXT_two",
    },
    "1": {
        'initial_state': np.zeros((64*64)),
        'model': model_configs['0'],
        'gama': 1e-4,
        'parameters':{'c1': 1, 'c2': 1, 'c3': 1, "p": 0.5, "q":1.5},
        'algorithm': "FXT_three",
    },
    "2": {
        'initial_state': np.zeros((128*128)),
        'model': model_configs['1'],
        'gama': 1,
        'parameters':{'c1': 10, 'c2': 10, "p": 0.5, "q":1.5, "u":1, "v":1, "initia_time": 0.01},
        'algorithm': "FXT_varying",
    },
    "2_1": {
        'initial_state': np.zeros((128*128)),
        'model': model_configs['1'],
        'gama': 1,
        'parameters':{'c1': 10, 'c2': 10, "p": 1, "q":1},
        'algorithm': "FXT_two",
    },
    "2_2": {
        'initial_state': np.zeros((128*128)),
        'model': model_configs['1'],
        'gama': 1,
        'parameters':{'c1': 0, 'c2': 10, "p": 0.5, "q":1.5},
        'algorithm': "FXT_two",
    },
    "2_4": {
        'initial_state': np.zeros((128*128)),
        'model': model_configs['1'],
        'gama': 1,
        'parameters':{'c1': 10, 'c2': 10, "p": 0.5, "q":1.5},
        'algorithm': "FXT_two",
    },
    "2_3": {
        'initial_state': np.zeros((128*128)),
        'model': model_configs['1'],
        'gama': 1,
        'parameters':{'c1': 10, 'c2': 10, 'c3': 10, "p": 0.5, "q":1.5},
        'algorithm': "FXT_three",
    },
}


class ImageRecover(Base):
    
    def __init__(self) -> None:
        super().__init__()
    
    def generate_blur_matrix(self, image, kernel, output_size):
        blur_matrix = np.zeros((output_size, image.size))
        padded_margin =  int(kernel.shape[0]/2)
        for i in range(image.shape[0]):  
            for j in range(image.shape[1]):
                row_index = i * (image.shape[1]) + j  
                for ki in range(kernel.shape[0]):  
                    for kj in range(kernel.shape[1]):
                        margin_i = i - padded_margin
                        margin_j = j - padded_margin
                        if margin_i + ki >=0 and margin_j + kj>= 0:
                            if margin_i + ki < image.shape[0] and margin_j + kj < image.shape[1]:
                                blur_matrix[row_index, (margin_i + ki) * image.shape[1] + (margin_j + kj)] = kernel[ki, kj]  
        
        return blur_matrix
        
        
    
    def set_blurred_info(self):
        image = self.model['image']
        kernel = self.model['kernel']
        padded_margin =  int(kernel.shape[0]/2)
        # padded_image = cv.copyMakeBorder(image, padded_margin, padded_margin, padded_margin, padded_margin, cv.BORDER_REPLICATE)
        output_size = (image.shape[0]) * (image.shape[1])
        
        self.blur_matrix = self.generate_blur_matrix(image, kernel, output_size)
        # self.blur_matrix = np.zeros((output_size, image.size))
        
        # output_size = (padded_image.shape[0]) * (padded_image.shape[1])
        # self.blur_matrix_padded = self.generate_blur_matrix(padded_image, kernel, output_size)
        
        # padded_image_vector = padded_image.flatten()
        # self.blurred_image = np.dot(self.blur_matrix_padded, padded_image_vector)
        # self.blurred_image = self.blurred_image.reshape((padded_image.shape[0] , padded_image.shape[1]))[1:-1, 1:-1]
        # self.blurred_image = self.blurred_image.flatten()
        
        # for i in range(image.shape[0]):  
        #     for j in range(image.shape[1]):  
        #         row_index = i * (image.shape[1]) + j  
        #         for ki in range(kernel.shape[0]):  
        #             for kj in range(kernel.shape[1]):
        #                 margin_i = i - padded_margin
        #                 margin_j = j - padded_margin
        #                 if margin_i + ki >=0 and margin_j + kj>= 0:
        #                     if margin_i + ki < image.shape[0] and margin_j + kj < image.shape[1]:  
        #                         self.blur_matrix[row_index, (margin_i + ki) * image.shape[1] + (margin_j + kj)] = kernel[ki, kj]  
        
        image_vector = self.model['image'].flatten()
        # padded_image_vector = padded_image.flatten()
        # self.blurred_image = np.dot(self.blur_matrix_padded, padded_image_vector)
        self.blurred_image = np.dot(self.blur_matrix, image_vector)
        
    
    def load_config(self, index, time_delta):
        self.config_index = index
        self.config = configs[index]
        self.model = self.config['model']
        self.state = self.config['initial_state']
        self.time_delta = time_delta
        self.set_blurred_info()
        self.records['global']['algorithm'] = self.config['algorithm']
        self.records['global']['image'] = self.model['image']
        self.records['global']['kernel'] = self.model['kernel']
        self.records['global']['blurred_image'] = self.blurred_image.reshape((self.model['image'].shape[0] , self.model['image'].shape[1]))
        # self.config['gama'] = 1/ (np.linalg.norm(self.blur_matrix, ord=2)**2)
                
    def update(self):
        self.optimizer.set_time_point(self.time_point)
        P_Q = self.blurred_image
        diff_value = (self.blur_matrix @ self.state) - P_Q
        # self.records['data'].append({"time": float(self.count)*self.time_delta, "state":list(self.state), "diffvalue": np.linalg.norm(diff_value)})
        u = self.state - self.config['gama']*(self.blur_matrix.T@diff_value)
        P_C = self.model['C'].project(u)
        phi = (self.state - P_C)
        self.state = np.array(self.state) + self.time_delta * self.optimizer.caculate(phi)
        
        
        if self.count % 10 == 0:
            self.records['data'].append({"time": float(self.time_point), "state":list(self.state), "P_Q": list(P_Q), "diffvalue": np.linalg.norm(diff_value)})

        if self.count % 100 == 0:
            print(np.linalg.norm(diff_value))
            recovered_img = self.state.reshape(self.model['image'].shape[0], self.model['image'].shape[1]).astype(np.uint8)
            cv.imwrite("./output/image_recover/{}/fig_{}.png".format(self.config_index, self.count), recovered_img)
            cv.imwrite("./output/image_recover/{}/origin.png".format(self.config_index, self.count), self.model['image'].astype(np.uint8))
            cv.imwrite("./output/image_recover/{}/blurred.png".format(self.config_index), self.records['global']['blurred_image'].astype(np.uint8)[1:-1, 1:-1])
        
        self.count += 1
        self.time_point += self.time_delta
        
        if self.count % 130 == 0 :
            self.time_delta = self.time_delta*2
        
        

    def save(self):
        folder_path = "./output/image_recover/{}".format(self.config_index)
        if not os.path.exists(folder_path):
            os.mkdir("./output/image_recover/{}".format(self.config_index))
        with open("{}/result.txt".format(folder_path), 'w') as f:
            f.write(json.dumps(self.records, cls=NpEncoder))
        with open("{}/result_finally_value.txt".format(folder_path), 'w') as f:
            f.write(json.dumps(self.records["data"][-10:], cls=NpEncoder))

    
    def apply_optimizer(self, optimizer):
        return super().apply_optimizer(optimizer)
