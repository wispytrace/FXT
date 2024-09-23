class Base:

    def __init__(self) -> None:
        self.time_point = 0

    def get_description(self):
        return self.DESC

    def set_time_point(self, time_point):
        self.time_point = time_point 
    
    def caculate(self):
        pass
    