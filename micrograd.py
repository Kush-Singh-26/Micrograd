import math

class Tensor:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self.grad = 0.0
    self._backprop = lambda: None

  def __repr__(self):
    return f"Tensor(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), '+')

    def _backprop():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backprop = _backprop

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)   # to allow opertations like (a * 2) or (a + 1)
    out = Tensor(self.data * other.data, (self, other), '*')

    def _backprop():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backprop = _backprop

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Tensor(t, (self, ), 'tanh')

    def _backprop():
      self.grad += (1 - t**2) * out.grad
    out._backprop = _backprop

    return out
  
  def relu(self):
    out = Tensor(0 if self.data < 0 else self.data, (self, ), 'ReLU')

    def _backprop():
      self.grad += (out.data > 0) * out.grad
    out._backprop = _backprop

    return out

  def backprop(self):
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)    # builds the topoloical graph

    self.grad = 1.0
    for node in reversed(topo):
      node._backprop()

  def __radd__(self, other):    # to perform operations like 2 + a
    return self + other

  def __rmul__(self, other):
    return self * other

  def exp(self):
    x = self.data
    out = Tensor(math.exp(x), (self, ), 'exp')

    def _backprop():
      self.grad += out.data * out.grad

    out._backprop = _backprop

    return out

  def pow(self, other):
    assert isinstance(other, (int, float)), "only float or int can be exponent"
    out = Tensor(self.data**other, (self,), f'**{other}')

    def _backprop():
      self.grad += (other * self.data ** (other -1)) * out.grad
    out._backprop = _backprop

    return out

  def _truediv__(self,other):
    return self * other**-1

  def __neg__(self):
    return self * -1
  
  def __sub__(self, other):
    return self + (-other)
  
  def __rsub__(self, other):
    return other + (-self)
  
  def __rtruediv__(self, other):
    return other * self**-1