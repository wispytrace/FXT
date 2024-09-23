from .base import *
import numpy as np
import json
import os


model_configs = {
    "0":{
        'A': np.random.normal(1, 1, (5,5)),
        'C': L1Ball(constant=30),
        'e': Matrix.generate_normal_vector(0.1, 0.2, 5),
        # 'true_value': Matrix.generate_signal_vector(5, base_wave=[0.0, 1.0, 0.0, 1.0, 1.0]),
    },
    "1":{
        'A': Matrix.generate_normal_matrix(0, 1, (256, 512), step_size=10),
        'C': L1Ball(constant=10),
        'e': Matrix.generate_normal_vector(0, 1e-3, 256),
        'true_value': Matrix.generate_signal_vector(512, wave_desc=[{'v':1, 'l':12}, {'v':-1, 'l':60}, {'v':1, 'l':100}, {'v':1, 'l':150}, {'v':-1, 'l':180}, {'v':-1, 'l':250}, {'v':1, 'l':275}, {'v':1, 'l':300}, {'v':-1, 'l':400}, {'v':1, 'l':500}]),
    }
}

configs = {
    "0": {
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-3,
        'parameters':{'c1': 0.1, 'c2': 0.1, "p": 0.6, "q":1.4},
        'algorithm': "FXT_two",
    },
    "1": {
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-3,
        'parameters':{'c1': 0.1, 'c2': 0.1, 'c3': 0.1, "p": 0.6, "q":1.4},
        'algorithm': "FXT_three",
    },
    "2": {
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-3, 
        'parameters':{'c1': 0.1, 'c2': 0.1, "p": 0.6, "q":1.4, "u":1, "v":1, "initia_time": 10},
        'algorithm': "FXT_varying",
    },
    "3": {
        'initial_state': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        'model': model_configs['0'],
        'gama': 0.1,
        'parameters':{'c1': 100, 'c2': 100, 'c3': 0, "p": 0.2, "q":1.2},
        'algorithm': "FXT_two",
    },
    "4":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-5, 
        'parameters':{'c1': 0.001, 'c2': 0.001, "p": 0.6, "q":1.4, "u":1, "v":1, "initia_time": 0.01},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_varying",
    },
    "4_1":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-5, 
        'parameters':{'c1': 0.1, 'c2': 0.1, "p": 1, "q":1},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_two",
    },
    "4_2":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-5, 
        'parameters':{'c1': 0.1, 'c2': 0.1, "p": 1, "q":1},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_two",
    },
    "4_3":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-5, 
        'parameters':{'c1': 0.1, 'c2': 0.1, "p": 0.6, "q":1.4},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_two",
    },
    "4_4":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-6, 
        'parameters':{'c1': 0.1, 'c2': 0.1, 'c3': 0.1, "p": 0.6, "q":1.4},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_three",
    },
    "5":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, "p": 0.6, "q":1.4, "u":1, "v":1, "initia_time": 0.01},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_varying",
    },
    "5_1":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 0, "p": 1, "q":1},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_two",
    },
    "5_2":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 0, "p": 0.6, "q":1},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_two",
    },
    "5_3":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, "p": 0.6, "q":1.4},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_two",
    },
    "5_4":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, 'c3': 5, "p": 0.6, "q":1.4},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_three",
    },
    "6":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, "p": 0.6, "q":1.4, "u":1, "v":1, "initia_time": 0.01},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_varying",
    },
    "6_1":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, "p": 0.3, "q":1.4, "u":1, "v":1, "initia_time": 0.01},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_varying",
    },
    "6_2":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, "p": 0.6, "q":1.2, "u":1, "v":1, "initia_time": 0.01},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_varying",
    },
    "6_3":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, "p": 0.6, "q":5, "u":1, "v":1, "initia_time": 0.01},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_varying",
    },
    "6_4":{
        'initial_state': np.zeros(512),
        'model': model_configs['1'],
        'gama': 1e-2, 
        'parameters':{'c1': 5, 'c2': 5, "p": 0.6, "q":1.4, "u":1, "v":1, "initia_time": 0.001},
        'norm': 2,
        'alhpa': 0,
        'algorithm': "FXT_varying",
    },
}

class SignalRecover(Base):
    
    def __init__(self) -> None:
        super().__init__()

    def load_config(self, index, time_delta):
        self.config_index = index
        self.config = configs[index]
        self.model = self.config['model']
        self.state = self.config['initial_state']
        self.records['global']['algorithm'] = self.config['algorithm'] + self.get_norm_info()
        self.records['global']['e'] = self.model['e']
        self.records['global']['true_value'] = self.model['true_value']
        self.records['global']['observed_value'] = self.model['A']@self.model['true_value'] + self.model['e']
        self.config['gama'] = 1/ (np.linalg.norm(self.model['A'], ord=2)**2)
        self.time_delta = time_delta
    
    def get_norm_info(self):
        
        if 'norm' in self.config.keys():
            return ' L'+ str(self.config['norm']) +'-norm'
        else:
            return ''
    
    def get_norm_value(self, x):
        if 'norm' in self.config.keys():
            signs = np.sign(x)
            abs_powered = np.abs(x) ** (self.config['norm']-1)  
            return abs_powered * signs * self.config['alhpa']
        else:
            return 0 * x
        
     
    
    def update(self):
        self.optimizer.set_time_point(float(self.count)*self.time_delta)
        P_Q = self.records['global']['observed_value']
        diff_value = (self.model['A'] @ self.state) - P_Q
        abs_value = self.state - self.model['true_value']
        self.records['data'].append({"time": float(self.count)*self.time_delta, "state":list(self.state), "diffvalue": np.linalg.norm(abs_value)})
        u = self.state - self.config['gama']*(self.model['A'].T@diff_value)
        P_C = self.model['C'].project(u)
        phi = (self.state - P_C)
        self.state +=  self.time_delta * self.optimizer.caculate(phi)
        self.count += 1


    def save(self):
        folder_path = "./output/signal_recover/{}".format(self.config_index)
        if not os.path.exists(folder_path):
            os.mkdir("./output/signal_recover/{}".format(self.config_index))
        with open("{}/result.txt".format(folder_path), 'w') as f:
            f.write(json.dumps(self.records, cls=NpEncoder))

    def apply_optimizer(self, optimizer):
        super().apply_optimizer(optimizer)

