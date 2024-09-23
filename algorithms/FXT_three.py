from .base import Base
import numpy as np

class Algorithm(Base):
    DESC = "A new fixed-time stability of neural network to solve split convex feasibility problems"

    def __init__(self) -> None:
        super().__init__()
        self.p = 0
        self.q = 0
        self.c1 = 0
        self.c2 = 0 
        self.c3 = 0
    
    def set_parameters(self, parameters):
        self.p = parameters['p']
        self.q = parameters['q']
        self.c1 = parameters['c1']
        self.c2 = parameters['c2']
        self.c3 = parameters['c3']

    def caculate(self, phi):
        norm = max(np.linalg.norm(phi), 1e-10)
        rho1 = np.power(norm, 1 - self.p)
        rho2 = np.power(norm, 1 - self.q)
        rho3 = norm
        if np.fabs(rho1) < 1e-10:
            rho1 = 0
        else:
            rho1 = 1/rho1
            
        if np.fabs(rho2) < 1e-10:
            rho2 = 0
        else:
            rho2 = 1/rho2
            
        if np.fabs(rho3) < 1e-10:
            rho3 = 0
        else:
            rho3 = 1/rho3
        
        deviation = -(self.c1*phi*rho1 + self.c2*phi*rho2 + self.c3*phi*rho3)
        return deviation

    def get_settle_time_2012(self, c1, c2, c3, p, q):
        pass

    def get_settle_time_2021(self, c1, c2, c3, p, q):
        # Distributed Fixed-Time Optimization in Economic Dispatch Over Directed Networks
        pass
        
        