class Car:

    def __init__(self, initial_state, model):
        self.state = initial_state
        self.model = model

    def step(self, u, dt):
        self.state = self.model.step(self.state, u, dt)

    @property
    def x(self):
        return self.state[0]
    
    @property
    def y(self):
        return self.state[1]

    @property
    def position(self):
        return self.state[:2]
    
    @property
    def heading(self):
        return self.state[2]
    
    @property
    def velocity(self):
        return (self.state[3]**2 + self.state[4]**2)**0.5
    