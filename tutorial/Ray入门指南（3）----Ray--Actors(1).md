本篇主要介绍actor的定义、创建，应用及内部的运行机制。**actor之间的调度是并行的，同一个actor上的运行是串行的。** actor是一个有状态的worker，其在python脚本中的表示形式就是把一个类定义为远程类。着重列举了actor的几个应用实例---actor并行和串行运行、用actor进行强化学习状态共享、在神经网络中应用和actor句柄的转递。其中的神经网络的例子可以作为一个典型的事例---tensorflow代码在在Ray中运行神经网络事例。最后actor的限制问题。
## Actors
远程(remote) 函数在Ray中是功能性和实用性函数。将自己限制在远程函数中可以实现分布式函数编程，这对于许多用例来说都是很好的，但是在实践中却受到了一定的限制。
Ray使用actor扩展了数据流模型。actor本质上是一个有状态的worker(或服务)。当实例化一个新actor时，将创建一个新worker，并将acto的方法调度到该特定worker上，并且可以访问该worker并更改其状态。
actor的形式之一是把一个类前面加上`@ray.remote`，这个类就变成一个actor，还有其他的定义actor的形式把一个类作为参数传到`ray.remote(类Ａ)`。


假设我们已经启动Ray。

    import ray
    ray.init()

## 定义和创建一个actor

下面的例子，用`ray.remote`装饰一个`Counter`类作为一个actor。

```
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value
```
可是通过`Counter.remote()`来实例化这个actor 。

    a1 = Counter.remote()
    a2 = Counter.remote()

实例化参与者时，将发生以下事件。

 1. 选择集群中的一个节点，并在该节点上创建一个工作进程(由该节点上的本地调度程序创建)，以便调用对参与者的方法。
 2. 在worker上创建一个Counter对象，并且运行Counter构造器。

## actor的应用--并行和串行
我们可以通过调用actor的方法来调度它的任务。

```
a1.increment.remote()  # ray.get returns 1
a2.increment.remote()  # ray.get returns 1
```
当调用`a1.increment.remote()`时，有下面事件发生。

 1. 创建一个任务。
 2. 任务由驱动程序的本地调度程序直接分配给负责actor的本地调度程序。
 3. 一个对象ID被返回。
 
 通过`ray.get(id)`来获取这个对象ID检索真实值。
 类似地，我们调用`a2.increment.remote()`在第二个`Counter` actor上生成一个任务。
由于这两个任务运行在不同的actor上，所以它们可以并行执行(注意，只有actor方法在actor worker上调度，而常规远程函数不会)。

另一方面，在同一个`Counter`actor上调用的方法按调用顺序串行执行。因此，它们可以彼此共享状态，如下所示。

```
# 创建 Counter actors.
counters = [Counter.remote() for _ in range(10)]

# 这些任务时并行的
results = ray.get([c.increment.remote() for c in counters])
print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# 这些任务是顺序执行的，共享状态，因为这是同一个actor内部的调用。
results = ray.get([counters[0].increment.remote() for _ in range(5)])
print(results)  # prints [2, 3, 4, 5, 6]
```

## 一个典型的actor例子--强化学习状态共享
常见的模式是使用actor封装由外部库或服务管理的可变状态。
[Gym](https://gym.openai.com/)为测试和训练强化学习代理提供了一个与许多模拟环境的接口。这些模拟器是有状态的，使用这些模拟器的任务必须更改它们的状态。我们可以使用角色来封装这些模拟器的状态。

```
import gym

@ray.remote
class GymEnvironment(object):
    def __init__(self, name):
        self.env = gym.make(name)
        self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
```
然后，我们可以实例化一个actor并在该参与者上调度任务，如下所示。

```
pong = GymEnvironment.remote("Pong-v0")
pong.step.remote(0)  # Take action 0 in the simulator.
```
这个例子主要说明**同一个actor中任务是顺序执行的，可以共享状态。**

## 在actor上使用GPUs---神经网络中应用
一个常见的用例是actor包含一个神经网络。例如，假设我们导入了Tensorflow并创建了一种构造神经网络的方法。

```
import tensorflow as tf

def construct_network():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y_, train_step, accuracy
```
然后，我们可以为这个网络定义一个actor，如下所示。

```
import os

# 定义在gpu上运行的参与者。
# 如果没有gpu，那么只需使用ray.remote没有任何参数，也没有括号。
@ray.remote(num_gpus=1)
class NeuralNetOnGPU(object):
    def __init__(self):
        # 设置一个环境变量来告诉TensorFlow使用哪个gpu。
        # 注意，这必须在调用tf.Session之前完成。
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in ray.get_gpu_ids()])
        with tf.Graph().as_default():
            with tf.device("/gpu:0"):
                self.x, self.y_, self.train_step, self.accuracy = construct_network()
                # Allow this to run on CPUs if there aren't any GPUs.
                config = tf.ConfigProto(allow_soft_placement=True)
                self.sess = tf.Session(config=config)
                # Initialize the network.
                init = tf.global_variables_initializer()
                self.sess.run(init)
```
为了表示一个actor需要一个GPU，我们将`num_gpu =1`传入`ray.remote`。请注意，为了使其工作，Ray必须使用一些gpu启动，例如，通过`Ray .init(num_gpu =2)`。否则，当您试图用`NeuralNetOnGPU.remote()`实例化GPU版本时，将抛出一个异常，说明系统中没有足够的GPU。
当创建actor时，可以通过通过`ray.get_gpu_ids()`获得gpu ID列表。
这是一个整数列表，比如[]、[1]或[2,5,6]。因为我们传入了`ray.remote(num_gpu =1)`，所以这个列表的长度为1。

我们可以把这些放在一起，如下。

```
import os
import ray
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

ray.init(num_gpus=8)

def construct_network():
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y_, train_step, accuracy

@ray.remote(num_gpus=1)
class NeuralNetOnGPU(object):
    def __init__(self, mnist_data):
        self.mnist = mnist_data
        # Set an environment variable to tell TensorFlow which GPUs to use. Note
        # that this must be done before the call to tf.Session.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in ray.get_gpu_ids()])
        with tf.Graph().as_default():
            with tf.device("/gpu:0"):
                self.x, self.y_, self.train_step, self.accuracy = construct_network()
                # Allow this to run on CPUs if there aren't any GPUs.
                config = tf.ConfigProto(allow_soft_placement=True)
                self.sess = tf.Session(config=config)
                # Initialize the network.
                init = tf.global_variables_initializer()
                self.sess.run(init)

    def train(self, num_steps):
        for _ in range(num_steps):
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

    def get_accuracy(self):
        return self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images,
                                                       self.y_: self.mnist.test.labels})


#加载MNIST数据集，并告诉Ray如何序列化定制类。
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Create the actor.
nn = NeuralNetOnGPU.remote(mnist)

# Run a few steps of training and print the accuracy.
nn.train.remote(100)
accuracy = ray.get(nn.get_accuracy.remote())
print("Accuracy is {}.".format(accuracy))
```
此例子是除了actor使用GPU，还是一个**神经网络在ray中应用的典型范例。**

## 例子——传递actor句柄
参与者句柄可以传递到其他任务中。要查看此示例，请查看 

 - [ ] [异步参数服务器示例](http://ray.readthedocs.io/en/latest/example-parameter-server.html)

用一个简单的例子来说明这一点，考虑一个简单的actor定义。该功能目前还处于试验阶段，并受到以下描述的限制。

```
@ray.remote
class Counter(object):
    def __init__(self):
        self.counter = 0

    def inc(self):
        self.counter += 1

    def get_counter(self):
        return self.counter
```

我们可以定义使用actor句柄的远程函数(或actor方法)。

```
@ray.remote
def f(counter):
    while True:
        counter.inc.remote()
```
如果我们实例化一个actor，我们可以将句柄传递给各种任务。

```
counter = Counter.remote()

# Start some tasks that use the actor.
[f.remote(counter) for _ in range(4)]

# Print the counter value.
for _ in range(10):
    print(ray.get(counter.get_counter.remote()))
```


## 当前actor的限制
有以下几个问题：

 1. **actor生命周期的管理**：目前，当一个actor的原始actor句柄超出范围时，会在该actor上安排一个任务，该任务会终止actor进程（这个新任务将在所有先前任务完成运行后运行）。如果原始actor处理程序超出范围，但actor仍被正在传递actor处理的任务使用，则这可能就是一个问题。
 2. **返回actor句柄**：actor句柄目前无法从远程函数或acror方法返回。同样,`ray.put`不能调用actor句柄。
 3. **重构被驱逐的actor对象**：如果对一个由actor方法创建的被驱逐的对象调用`ray.get`, Ray目前不会重构该对象。有关更多信息，请参阅有关-----------[容错的文档](https://blog.csdn.net/weixin_43255962/article/details/89684608)。
 4. **确定重建丢失的actor**：如果某个actor由于节点故障而丢失，则按照初始执行的顺序在新节点上重构该actor。然而，同时调度到actor上的新任务可能在重新执行的任务之间执行。如果您的应用程序对状态一致性有严格的要求，那么这可能就是个问题。


此篇主要参考Ray官网，如有错误，请阅读者提出指正，谢谢！
原英文链接：https://ray.readthedocs.io/en/latest/actors.html


