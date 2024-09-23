import numpy as np
from scipy.optimize import fsolve
import json
import copy
import random
random.seed(666)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)

class Base:

    def __init__(self) -> None:
        self.records = {"global": {}, "data": []}
        self.count = 0
        self.time_point = 0
    
    def save(self):
        pass
        
    def apply_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.set_parameters(self.config['parameters'])

    def update(self, time_delta):
        pass


class ConstraintBase:
    
    def __init__(self) -> None:
        pass
    
    def get_value(self):
        pass
    
    def project(self):
        pass

class SetLevel(ConstraintBase):
    "l<y<u"
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high

    def project(self, x):
        for i in range(len(x)):
            x[i] = max(min(x[i], self.high[i]), self.low[i])
        return x 

class LinearBall(ConstraintBase):
    "ax^2 + by + cz - constant <= 0 "
    def __init__(self, series, order, constant) -> None:
        self.series = series
        self.order = order
        self.constant = constant
    
    def get_value(self, x):
        value = 0
        for i in range(len(self.series)):
            value += self.series[i]*np.power(x[i], self.order[i])
        return value - self.constant
    
    def equation(self, lambda_star, x):
        value = 0
        for i in range(len(self.series)):
            lower = -np.inf
            if self.order[i] % 2 == 0:
                x_hat = np.power(x[i], self.order[i])
                lower = 0
            else:
                x_hat = x[i]
            z = x_hat - lambda_star*self.series[i]
            value += self.series[i]*(max(lower, x_hat - lambda_star*self.series[i]))
                
        return value - self.constant
    
    def project(self, x):
        x = np.array(x, dtype=float)
        if self.get_value(x) <= 0:
            return x
        else:
            lambda_star = fsolve(self.equation, 1, args=(copy.deepcopy(x), ))
            for i in range(len(self.series)):
                lower = -np.inf
                if self.order[i] % 2 == 0:
                    lower = 0
                    x_hat = np.power(x[i], self.order[i])
                else:
                    x_hat = x[i]
                x[i] = np.power(max(x_hat - lambda_star*self.series[i], lower), 1/self.order[i]) 
        
        return x


class L1Ball(ConstraintBase):
    "|x|_1 <= c "
    def __init__(self, constant) -> None:
        self.constant = constant
    
    def get_value(self, x):
        value = np.fabs(x).sum()
        return value - self.constant
    
    def equation(self, lambda_star, x):
        y = copy.deepcopy(x)
        u = np.fabs(y)
        for i in range(len(x)):
            y[i] = max(0, u[i] - lambda_star) * np.sign(y[i])
        value = np.fabs(y).sum() - self.constant
        return value
        
    
    def project(self, x):
        x = np.array(x, dtype=float)
        if self.get_value(x) <= 0:
            return x
        else:
            y = copy.deepcopy(x)
            u = np.fabs(y)
            initial_value = (np.fabs(x).sum()- self.constant)/len(x)
            lambda_star = fsolve(self.equation, initial_value, args=(x,))
            for i in range(len(x)):
                y[i] = max(0, u[i] - lambda_star) * np.sign(y[i])
            return y

class L2Ball(ConstraintBase):
    "||x-r||<c"
    def __init__(self, radius, constant) -> None:
        self.constant = constant
        self.radius = radius
    
    def get_value(self, x):
        value = np.linalg.norm(x-self.radius) - self.constant
        return value
    

    def project(self, x):
        x = np.array(x, dtype=float)
        if self.get_value(x) <= 0:
            return x
        else:
            x = self.radius + self.constant/(max(np.linalg.norm(x-self.radius), self.constant))*(x-self.radius)
            return x



class Matrix():
    
    @staticmethod
    def generate_normal_matrix(mean, std_dev, shape, step_size=10):
        # gaussian_vector = np.random.normal(loc=mean, scale=std_dev, size=step_size) 
        # gaussian_vector = gaussian_vector/np.sum(gaussian_vector)
        # matrix =  np.zeros(shape)
        # gap = shape[1] - shape[0]
        # ratio = int(shape[1] / shape[0])
        # for i in range(shape[0]):
        #     for j in range(step_size):
        #         if 2*i-int(step_size/2)+j >= shape[1] or 2*i-int(step_size/2)+j <0:
        #             continue
        #         matrix[i,2*i-int(step_size/2)+j] = gaussian_vector[j]
        matrix = np.random.randn(shape[0], shape[1]) 
        # matrix = np.random.normal(loc=mean, scale=std_dev, size=shape)

        return matrix
    
    @staticmethod
    def generate_normal_vector(min_value, max_value, shape):
        vector = np.zeros(shape)
        for i in range(shape):
            vector[i] = (max_value - min_value)/shape * i + min_value + random.random()*(max_value-min_value)

        return vector

    @staticmethod
    def generate_signal_vector(shape, wave_desc):
        # base_wave = []
        # for item in wave_desc:
        #     wave = [item['v']] * item['l']
        #     base_wave.extend(wave)
        vector = np.zeros(shape)
        # period = len(base_wave)
        for item in wave_desc:
            # wave = [item['v']] * item['l']
            vector[item['l']] = item['v']
        # for i in range(shape):
        #     vector[i] = base_wave[i%period]
        return vector
    
    @staticmethod
    def gaussian_kernel(size, sigma=1.0):  
        size = int((size-1)/2)  
        x, y = np.mgrid[-size:size+1, -size:size+1]  
        g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))  
        return g / g.sum()  