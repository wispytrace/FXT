from .base import *
import numpy as np
import json
import os

model_configs = {
    "0":{
        'A': np.array([[2, -1 ,3],[4, 2, 5],[2, 0, 2]]),
        'B': np.eye(3), 
        'C': LinearBall(series=[1, 1, 2], order=[1, 2, 1], constant=0),
        'Q': LinearBall(series=[1, 1, -1], order=[2, 1, 1], constant=0),
    }
}

configs = {
    "0": {
        'model': model_configs['0'],
        'initial_state': np.array([1, 1, 1]),
        'N': 3,
        'gama': 0.1,
        'parameters':{'c1': 0.2, 'c2': 0.2, "p": 0.2, "q":1.2},
        'algorithm': "FXT_two",
    },
    "1": {
        'model': model_configs['0'],
        'initial_state': np.array([1, 1, 1]),
        'N': 3,
        'gama': 0.1,
        'parameters':{'c1': 0.2, 'c2': 0.2, 'c3': 2, "p": 0.2, "q":1.2},
        'algorithm': "FXT_three",
    },
    "2": {
        'model': model_configs['0'],
        'initial_state': np.array([1, 1, 1]),
        'N': 3,
        'gama': 0.1,
        'parameters':{'c1': 0.2, 'c2': 0.2,  "p": 0.2, "q":1.2, "u":1, "v":1, "initia_time": 10},
        'algorithm': "FXT_varying",
    },
    "2_1": {
        'model': model_configs['0'],
        'initial_state': np.array([1, 1, 1]),
        'N': 3,
        'gama': 0.1,
        'parameters':{'c1': 0.2, 'c2': 0.2,  "p": 1, "q":1},
        'algorithm': "FXT_two",
    },
    "2_2": {
        'model': model_configs['0'],
        'initial_state': np.array([1, 1, 1]),
        'N': 3,
        'gama': 0.1,
        'parameters':{'c1': 0.2, 'c2': 0.2,  "p": 0.2, "q":1},
        'algorithm': "FXT_two",
    },
    "2_3": {
        'model': model_configs['0'],
        'initial_state': np.array([1, 1, 1]),
        'N': 3,
        'gama': 0.1,
        'parameters':{'c1': 0.2, 'c2': 0.2, 'c3': 2, "p": 0.2, "q":1.2},
        'algorithm': "FXT_three",
    },
}


class SCFP(Base):
    
    def __init__(self) -> None:
        super().__init__()

    def load_config(self, index, time_delta):
        self.config_index = index
        self.config = configs[index]
        self.model = self.config['model']
        self.state = self.config['initial_state']
        self.time_delta = time_delta
        self.records['global']['algorithm'] = self.config['algorithm']

    def memory_to_list(self, memory):
        listed_memory = {}
        for k, v in memory.items():
            listed_memory[k] = v.tolist()
        return listed_memory
    
    def update(self):
        self.optimizer.set_time_point(float(self.count)*self.time_delta)
        P_Q = self.model['Q'].project(self.model['A']@self.state)
        diff_value = (self.model['A'] @ self.state) - P_Q
        self.records['data'].append({"time": float(self.count)*self.time_delta, "state":list(self.state), "P_Q": list(P_Q), "diffvalue": np.linalg.norm(diff_value)})
        u = self.state - self.config['gama']*(self.model['A'].T @ diff_value)
        P_C = self.model['C'].project(u)
        phi = (self.state - P_C)
        self.state = np.array(self.state) + self.time_delta * self.optimizer.caculate(phi)
        self.count += 1

    def save(self):
        folder_path = "./output/SCFP/{}".format(self.config_index)
        if not os.path.exists(folder_path):
            os.mkdir("./output/SCFP/{}".format(self.config_index))
        with open("{}/result.txt".format(folder_path), 'w') as f:
            f.write(json.dumps(self.records, cls=NpEncoder))
            
    def apply_optimizer(self, optimizer):
        super().apply_optimizer(optimizer)

