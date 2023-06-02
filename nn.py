class Value:
    def __init__(self, data):
        self.data  = data
        self.grad = 0

    def __add__(self, other: 'Value'):
        val = Value(self.data + other.data)
        return val

    def calc_grad(self, upward_grad):
        self
