本篇仅初步简单介绍Ray中的重要训练框架Tune（可伸缩的并行的超参搜索框架）的特点和简单的入门操作。详细的[tune用户指南请看下一篇](https://blog.csdn.net/weixin_43255962/article/details/89012548)。tune的主要应用是在深度学习和深度强化学习中。可以在GitHub上找到[Tune代码](https://github.com/ray-project/ray/tree/master/python/ray/tune)，要开始使用Tune，需要一定的储备知识，深度学习，Keras库，tensorflow等强化学习的知识。

## 特点

 - 支持任何深度学习框架，包括PyTorch、TensorFlow和Keras。
 - 选择可伸缩的超参数和模型搜索算法，如**Population Based Training (PBT)**、**Median Stopping Rule**、**HyperBand**。
 - 混合并匹配不同的超参数优化方法——例如**HyperOpt with HyperBand** 或 **Nevergrad with HyperBand**.
 - 使用**TensorBoard,** **parallel coordinates (Plot.ly),** 和 **rllab’s VisKit**可视化结果。
 - 在不更改代码的情况下扩展到在大型分布式集群上运行。
 - 使用Tune的资源感知调度，对具有GPU要求或算法的模型进行并行化培训，这些模型本身可以并行和分布，

## 入门指南
#### 入门指南
首先安装安装ray来导入Tune。

    pip install ray 

#### 快速启动
本例使用Tune在一个神经网络训练函数上运行一个小型网格搜索，报告命令行上的状态，直到达到`mean_accuracy >= 99`的条件停止。**Tune适用于任何深度学习框架。**
Tune使用Ray作为后端，因此我们将首先导入和初始化Ray。（注意tune是在Ray中的，需要额外导入）

```
# 导包
import ray
from ray import tune
# 启动Ray
ray.init()
```
对于要调优的函数，传入一个`reporter`对象:

```
def train_func(config, reporter):  # add a reporter arg
     model = ( ... )
     optimizer = SGD(model.parameters(),
                     momentum=config["momentum"])
     dataset = ( ... )

     for idx, (data, target) in enumerate(dataset):
         accuracy = model.fit(data, target)
         reporter(mean_accuracy=accuracy) # report metrics
```
最后，配置您的搜索算法并在您的Ray集群上执行下面代码:

```
all_trials = tune.run(
    train_func,
    name="quick-start",
    stop={"mean_accuracy": 99},
    config={"momentum": tune.grid_search([0.1, 0.2])}
)
```
Tunr可以在Ray的任何地方使用，例如在你的笔记本上的python脚本中使用，或者在自动伸缩集群中实现大规模并行。

## 引用Tune

如果Tune对你的学术研究有帮助，可以引用[Tune的论文](https://arxiv.org/abs/1807.05118) 。这里有一个bibtex的例子:

```
@article{liaw2018tune,
    title={Tune: A Research Platform for Distributed Model Selection and Training},
    author={Liaw, Richard and Liang, Eric and Nishihara, Robert
            and Moritz, Philipp and Gonzalez, Joseph E and Stoica, Ion},
    journal={arXiv preprint arXiv:1807.05118},
    year={2018}
}
```
此篇主要参考Ray官网，如有错误，请阅读者提出指正，谢谢！
原英文链接：https://ray.readthedocs.io/en/latest/tune.html
 Ray----Tune(2):Tune的用户指南 ：https://blog.csdn.net/weixin_43255962/article/details/89012548
