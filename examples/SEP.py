from .base import *
import numpy as np
import json
import os

model_configs = {
    "0":{
        'x': np.array([2.0, 3.0]),
        'y': np.array([8.0, 1.0]),
        'A': np.array([[0.8738, 0.9352],[0.8642, 0.414]]),
        'B': np.array([[0.4027, 0.0916],[0.2568, 0.1888]]), 
        'C': L2Ball(radius=[6.9286, 0.5499], constant=59.8580),
        'Q': SetLevel(low=[11.6697, 16.1374], high=[98.0785, 80.7233]),
    }
}

configs = {
    "0": {
        'model': model_configs['0'],
        'gama': 0.1,
        'parameters':{'c1': 1, 'c2': 1, "p": 0.5, "q":1.5},
        'algorithm': "FXT_two",
    },
    "1": {
        'model': model_configs['0'],
        'gama': 0.1,
        'parameters':{'c1': 1, 'c2': 1, 'c3': 1, "p": 0.5, "q":1.5},
        'algorithm': "FXT_three",
    },
    "2": {
        'model': model_configs['0'],
        'gama': 0.2,
        'parameters':{'c1': 0.002, 'c2': 0.002,  "p": 0.5, "q":1.5, "u":1, "v":1, "initia_time": 0.9},
        'algorithm': "FXT_varying",
    },
    "2_1": {
        'model': model_configs['0'],
        'gama': 0.3,
        'parameters':{'c1': 0.2, 'c2': 0.2,  "p": 0.5, "q":1.5, "u":1, "v":1, "initia_time": 10},
        'algorithm': "FXT_varying",
    },
    "2_2": {
        'model': model_configs['0'],
        'gama': 0.3,
        'parameters':{'c1': 0.4, 'c2': 0.4,  "p": 0.5, "q":1.5, "u":1, "v":1, "initia_time": 20},
        'algorithm': "FXT_varying",
    },
    "2_3": {
        'model': model_configs['0'],
        'gama': 0.2,
        'parameters':{'c1': 0.002, 'c2': 0.002, "p": 1, "q":1},
        'algorithm': "FXT_two",
    },
    
}


class SEP(Base):
    
    def __init__(self) -> None:
        super().__init__()

    def load_config(self, index, time_delta):
        self.config_index = index
        self.config = configs[index]
        self.model = self.config['model']
        self.records['global']['algorithm'] = self.config['algorithm']
        self.x = self.model['x']
        self.y = self.model['y']
        self.time_delta = time_delta

    def test(self):
        x = np.array([6.9574, 0.5764])
        y = np.array([11.6699, 16.1375])
        diff_value = self.model['A']@x - self.model['B']@y
        print(diff_value)
    
    def memory_to_list(self, memory):
        listed_memory = {}
        for k, v in memory.items():
            listed_memory[k] = v.tolist()
        return listed_memory
    
    def update(self):
        # self.test()
        diff_value = self.model['A']@self.x - self.model['B']@self.y
        self.optimizer.set_time_point(float(self.count)*self.time_delta)
        self.records['data'].append({"time": float(self.count)*self.time_delta, "x":list(self.x), "y":list(self.y), "diffvalue": np.linalg.norm(diff_value)})
        P_C = self.model['C'].project(self.x - self.config['gama']*(self.model['A'].T@diff_value))
        P_Q = self.model['Q'].project(self.y + self.config['gama']*(self.model['A'].T@diff_value))
        phi_x = self.x - P_C
        phi_y = self.y - P_Q
        self.x += self.time_delta * self.optimizer.caculate(phi_x)
        self.y += self.time_delta * self.optimizer.caculate(phi_y)
        self.count += 1

    def save(self):
        folder_path = "./output/SEP/{}".format(self.config_index)
        os.makedirs("./output/SEP/{}".format(self.config_index), exist_ok=True)    
        with open("{}/result.txt".format(folder_path), 'w') as f:
            f.write(json.dumps(self.records, cls=NpEncoder))
            
    def apply_optimizer(self, optimizer):
        super().apply_optimizer(optimizer)

