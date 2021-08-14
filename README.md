# PyTorch-optimizer-test

### 为了更好的帮大家了解优化器，我们对PyTorch中的优化器进行了一个小测试

### 数据生成：
```python
a = torch.linspace(-1, 1, 1000)
# 升维操作
x = torch.unsqueeze(a, dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
```
### 数据分布曲线：
![](figures/figures.png)

### 网络结构定义
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x

```
### 下面这部分是测试图，纵坐标代表Loss，横坐标代表的是Step：

![](figures/myplot.png)

在上面的图片上，曲线下降的趋势和对应的steps代表了在这轮数据，模型下的收敛速度

#### Tips:
优化器的选择是需要根据模型进行改变的，不存在绝对的好坏之分，我们需要多进行一些测试。

### 需改进的地方：
SparseAdam，LBFGS这两个优化器没有可视化成功，后面需要继续进行改变
