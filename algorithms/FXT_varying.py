from .base import Base
import numpy as np

class Algorithm(Base):
    DESC = "A new fixed-time stability of neural network to solve split convex feasibility problems"

    def __init__(self) -> None:
        super().__init__()
        self.p = 0
        self.q = 0
        self.u = 0
        self.v = 0 
        self.c1 = 0
        self.c2 = 0
    
    def set_parameters(self, parameters):
        self.p = parameters['p']
        self.q = parameters['q']
        self.u = parameters['u']
        self.v = parameters['v']
        self.c1 = parameters['c1']
        self.c2 = parameters['c2']
        self.initia_time = parameters['initia_time']

    def caculate(self, phi):
        c1 = self.c1/np.power((self.time_point + self.initia_time), self.u)
        c2 = self.c2/np.power((self.time_point + self.initia_time), self.v)
        norm = max(np.linalg.norm(phi), 1e-10)
        rho1 = np.power(norm, 1 - self.p)
        rho2 = np.power(norm, 1 - self.q)
        if np.fabs(rho1) < 1e-10:
            rho1 = 0
        else:
            rho1 = 1/rho1
        if np.fabs(rho2) < 1e-10:
            rho2 = 0
        else:
            rho2 = 1/rho2
        
        deviation = -( c1*phi*rho1 + c2*phi*rho2)
        print(norm, phi[256], c1, c2)
        print(-(c1*phi*1/np.power(norm, 1 - self.p) + c2*phi*1/np.power(norm, 1 - 1.4))[256], deviation[256])
        
        return deviation

    def get_settle_time_2012(self, c1, c2, c3, p, q):
        pass

    def get_settle_time_2021(self, c1, c2, c3, p, q):
        # Distributed Fixed-Time Optimization in Economic Dispatch Over Directed Networks
        pass
        
        