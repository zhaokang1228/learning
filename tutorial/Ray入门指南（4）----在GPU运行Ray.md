gpu使用对于许多机器学习应用程序来说是至关重要的，特别是深度学习和深度强化学习。故本篇介绍Ray中的远程函数和actor通过`ray.remote`装饰器中指定它们的GPU需求，使用。
 使用gpu和cpu的方法是同样的，只是使用gpu需要指定，cpu是默认使用的。
 分为两步：１.ray启动的时候指定.如：ray.init(num_gpus=4,num_cpus=4)说明只有和此ray相连的机器总共使用４个gpu和4个cpu。
 	
	2.装饰的时候声明远程函数或者actor使用gpu和cpu的个数。如@ray.remote(num_gpus=1)　此远程函数或者actor使用１个gpu.

## 用GPU启动Ray
为了让远程函数和actor使用gpu, Ray必须知道有多少gpu可用。如果在单台机器上启动Ray，可以指定gpu的数量，如下所示。
```
ray.init(num_gpus=4)
```
如果不传递`num_gpus`参数，Ray将假定计算机上有0个gpu。
如果您正在使用`ray start`命令启动Ray，那么您可以使用`--number -gpu`参数来指示机器上的gpu数量。

```
ray start --head --num-gpus=4
```
**注意：**`num-gpus`不能传入比机器上实际GPU的大的值。如果`num-gpus`传入的GPU大于实际机器的数量，Ray会去调度指定的GPU数量，当这些任务试图去使用这些GPU的时候，就会报错。

## 在GPU上的远程函数
如果远程函数需要gpu，请在远程装饰器中指定所需的gpu数量。

```
@ray.remote(num_gpus=1)
def gpu_method():
    return "This function is allowed to use GPUs {}.".format(ray.get_gpu_ids())
```

在远程函数内部，对`ray.get_gpu_ids()`的调用将返回一个整数列表，该列表指示远程函数允许使用哪个gpu。

**注意：** 上面定义的函数`gpu_method`实际上并不使用任何gpu。Ray会把它安排在一台至少有一个GPU的机器上，并且在执行的时候会预留一个GPU给它，不过这取决于函数实际使用GPU的情况。这通常通过TensorFlow这样的外部库来完成。下面是一个实际使用gpu的例子。注意，要使这个示例工作，您需要安装TensorFlow的GPU版本。

```
import os
import tensorflow as tf

@ray.remote(num_gpus=1)
def gpu_method():
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids()))
    # Create a TensorFlow session. TensorFlow will restrict itself to use the
    # GPUs specified by the CUDA_VISIBLE_DEVICES environment variable.
    tf.Session()
```
**注意：** 对于实现`gpu_method`来说，忽略`ray.get_gpu_id`并使用计算机上的所有gpu当然是可能的。Ray并没有阻止这种情况的发生，这可能导致太多的worker同时使用相同的GPU。例如，如果未设置`CUDA_VISIBLE_DEVICES`环境变量，则TensorFlow将尝试使用机器上的所有gpu。

## GPU上的actor
在定义使用gpu的actor时，请在ray.remote装饰器中指出参与者实例所需的gpu数量。

```
@ray.remote(num_gpus=1)
class GPUActor(object):
    def __init__(self):
        return "This actor is allowed to use GPUs {}.".format(ray.get_gpu_ids())
```
当actor被创建时，gpu将在actor的生命周期中为该actor保留。

**注意：** Ray启动时的gpu数量必须至少与传递给`ray.remote`装饰器的gpu数量相同。否则，如果传入的数字大于传递给ray.init的数字，则在实例化actor时将引发异常。
下面是如何通过TensorFlow在actor中使用gpu的示例。

```
@ray.remote(num_gpus=1)
class GPUActor(object):
    def __init__(self):
        self.gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        # The call to tf.Session() will restrict TensorFlow to use the GPUs
        # specified in the CUDA_VISIBLE_DEVICES environment variable.
        self.sess = tf.Session()
```

## 问题

**注意：** 当前，当一个worker使用GPU去执行一个任务时，该任务可能会在GPU上分配内存，并且可能不会在任务执行完成时释放内存----这就会导致问题。更多[请参考问题](https://github.com/ray-project/ray/issues/616)。


此篇主要参考Ray官网，如有错误，请阅读者提出指正，谢谢！
原英文链接：https://ray.readthedocs.io/en/latest/using-ray-with-gpus.html
ray综述介绍：https://blog.csdn.net/weixin_43255962/article/details/88689665
ray-actor：https://blog.csdn.net/weixin_43255962/article/details/88854928 




