"""
MindSpore script error examples with multiple common failure scenarios.
"""
import mindspore as ms
from mindspore import nn, ops
import numpy as np

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

class DTypeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor(np.random.randn(10, 5).astype(np.float32)))
    
    def construct(self, x):
        return ops.matmul(x, self.weight)


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense = nn.Dense(10, 1)
    
    def construct(self, x):
        return self.dense(x)


if __name__ == "__main__":
    print("Running error scenarios...")
    
    dtype_net = DTypeNet()
    try:
        input_data = np.random.randn(2, 10)
        output = dtype_net(ms.Tensor(input_data))
    except Exception as e:
        print(f"Error 1 triggered: {e}")

    net = SimpleNet()
    try:
        x = ms.Tensor(np.random.randn(2, 10).astype(np.float32))
        y = net(x)
        y.backward()
    except Exception as e:
        print(f"Error 2 triggered: {e}")
