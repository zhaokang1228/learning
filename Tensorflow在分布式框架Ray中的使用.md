本篇主要介绍在Ray上使用tensorflow的最佳方法。当正在分布式环境中训练一个深度网络，可能需要在进程(或机器)之间传送您的深度网络(参数)。例如，可以在一台机器上更新您的模型，然后使用该模型在另一台机器上计算梯度。交付模型并不总是成功的，可能会出现一些错误信息。如，直接尝试pickle Tensorflow图会得到混合的结果。有些示例失败了，有些成功了(但是生成了非常大的字符串)。结果与其他pickle库类似。此外，创建Tensorflow图可能需要几十秒，因此序列化一个图并在另一个进程中重新创建它将是低效的。更好的解决方案是在开始时为每个worker创建相同的张Tensorflow图，然后只在worker之间传递权重。要查看更多使用TensorFlow的复杂示例，请查看[A3C、LBFGS、策略梯度](https://blog.csdn.net/weixin_43255962/article/details/89645053)和ResNet,或者参考[官网](https://ray.readthedocs.io/en/ray-0.5.3/example-resnet.html)。


假设我们有一个简单的网络定义(这个定义是从TensorFlow文档中修改的)。

```
import tensorflow as tf
import numpy as np

x_data = tf.placeholder(tf.float32, shape=[100])
y_data = tf.placeholder(tf.float32, shape=[100])

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
grads = optimizer.compute_gradients(loss)
train = optimizer.apply_gradients(grads)

init = tf.global_variables_initializer()
sess = tf.Session()
```

要提取权重并设置权重，可以使用以下方法。

```
# ray老版本
import ray.experimental.tf_utils
variables = ray.experimental.tf_utils.TensorFlowVariables(loss, sess)

# ray 新版本
import ray
variables = ray.experimental.TensorFlowVariables(loss, sess)
```
`TensorFlowVariables`对象提供了获取和设置权重以及收集模型中所有变量的方法。

现在我们可以使用这些方法来提取权重，并将它们放回网络中，如下所示。

```
# First initialize the weights.
sess.run(init)
# Get the weights
weights = variables.get_weights()  # Returns a dictionary of numpy arrays
# Set the weights
variables.set_weights(weights)
```

**注意:** 如果我们像下面这样使用`assign`方法设置权重，那么每个要`assign`的调用都会向图中添加一个节点，并且随着时间的推移，图会不可管理地变大。

```
w.assign(np.zeros(1))  # This adds a node to the graph every time you call it.
b.assign(np.zeros(1))  # This adds a node to the graph every time you call it.
```
下边是用ray解决这种方法的具体步骤。
## 1. 对权重求均值的完整例子
综上所述，我们首先将图形嵌入一个actor中。 在actor中，我们将使用`TensorFlowVariables`类的`get_weights`和`set_weights`方法。 然后，我们将使用这些方法在流程之间传递权重（作为映射到numpy数组的变量名称的字典），而不传送实际的TensorFlow图形，这些图形是更复杂的Python对象。

```
import tensorflow as tf
import numpy as np
import ray

ray.init()

BATCH_SIZE = 100
NUM_BATCHES = 1
NUM_ITERS = 201

class Network(object):
    def __init__(self, x, y):
        # Seed TensorFlow to make the script deterministic.　设置种子
        tf.set_random_seed(0)
        # Define the inputs.	定义输入
        self.x_data = tf.constant(x, dtype=tf.float32)
        self.y_data = tf.constant(y, dtype=tf.float32)
        # Define the weights and computation.　定义参数和计算
        w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = w * self.x_data + b
        # Define the loss.　定义损失
        self.loss = tf.reduce_mean(tf.square(y - self.y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        self.grads = optimizer.compute_gradients(self.loss)
        self.train = optimizer.apply_gradients(self.grads)
        # Define the weight initializer and session.　　初始化权重和session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        # Additional code for setting and getting the weights　添加获取和设置权重的代码
        self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
        # Return all of the data needed to use the network.　启动网络
        self.sess.run(init)

    # Define a remote function that trains the network for one step and returns the
    # new weights.
    # 定义训练网络的远程函数，返回新的权重
    def step(self, weights):
        # Set the weights in the network.　设置网络的权重
        self.variables.set_weights(weights)
        # Do one step of training.　执行一步训练
        self.sess.run(self.train)
        # Return the new weights.　返回新的权重
        return self.variables.get_weights()

	# 获得权重
    def get_weights(self):
        return self.variables.get_weights()

# Define a remote function for generating fake data.
# 定义远程函数去生成假的数据
@ray.remote(num_return_vals=2)
def generate_fake_x_y_data(num_data, seed=0):
    # Seed numpy to make the script deterministic.
    np.random.seed(seed)
    x = np.random.rand(num_data)
    y = x * 0.1 + 0.3
    return x, y

# Generate some training data.　生成训练数据
batch_ids = [generate_fake_x_y_data.remote(BATCH_SIZE, seed=i) for i in range(NUM_BATCHES)]
x_ids = [x_id for x_id, y_id in batch_ids]
y_ids = [y_id for x_id, y_id in batch_ids]
# Generate some test data.　生成测试数据
x_test, y_test = ray.get(generate_fake_x_y_data.remote(BATCH_SIZE, seed=NUM_BATCHES))

# Create actors to store the networks.　创建一个actor ，(形式ray.actor(类名))
remote_network = ray.remote(Network)
actor_list = [remote_network.remote(x_ids[i], y_ids[i]) for i in range(NUM_BATCHES)]

# Get initial weights of some actor.　获取一个actor的权重
weights = ray.get(actor_list[0].get_weights.remote())

# Do some steps of training.
for iteration in range(NUM_ITERS):
    # Put the weights in the object store. This is optional. We could instead pass
    # the variable weights directly into step.remote, in which case it would be
    # placed in the object store under the hood. However, in that case multiple
    # copies of the weights would be put in the object store, so this approach is
    # more efficient.
    # 上边总结一句话就是：把权重对象放到ray的对象存储中，效率更高。
    weights_id = ray.put(weights)
    # Call the remote function multiple times in parallel.　并行的调用多个远程函数
    new_weights_ids = [actor.step.remote(weights_id) for actor in actor_list]
    # Get all of the weights.　获取权重序列
    new_weights_list = ray.get(new_weights_ids)
    # Add up all the different weights. Each element of new_weights_list is a dict
    # of weights, and we want to add up these dicts component wise using the keys
    # of the first dict.
    weights = {variable: sum(weight_dict[variable] for weight_dict in new_weights_list) / NUM_BATCHES for variable in new_weights_list[0]}
    # Print the current weights. They should converge to roughly to the values 0.1
    # and 0.3 used in generate_fake_x_y_data.
    if iteration % 20 == 0:
        print("Iteration {}: weights are {}".format(iteration, weights))
```
## 2. 使用Ray和梯度进行并行训练
在某些情况下，您可能希望在您的网络上进行数据并行训练。我们使用上面的网络来说明如何在Ray中实现这一点。唯一的区别在于远程函数`step`和驱动程序代码。

在函数步骤中，我们运行grad操作而不是train操作来获得梯度。由于Tensorflow将梯度与元组中的变量配对，我们提取梯度以避免不必要的计算。

###  2.1 提取数值梯度
像下面这样的代码可以在远程函数中用于计算数值梯度。

```
x_values = [1] * 100
y_values = [2] * 100
numerical_grads = sess.run([grad[0] for grad in grads], feed_dict={x_data: x_values, y_data: y_values})
```

### 2.2 使用返回的梯度训练网络
通过将feed_dict中的符号梯度与数值梯度配对，我们可以更新网络。

```
# We can feed the gradient values in using the associated symbolic gradient
# operation defined in tensorflow.
feed_dict = {grad[0]: numerical_grad for (grad, numerical_grad) in zip(grads, numerical_grads)}
sess.run(train, feed_dict=feed_dict)
```
然后可以运行variables.get_weights()来查看网络的更新权重。

全文如下:

```
import tensorflow as tf
import numpy as np
import ray

ray.init()

BATCH_SIZE = 100
NUM_BATCHES = 1
NUM_ITERS = 201

class Network(object):
    def __init__(self, x, y):
        # Seed TensorFlow to make the script deterministic.
        tf.set_random_seed(0)
        # Define the inputs.
        x_data = tf.constant(x, dtype=tf.float32)
        y_data = tf.constant(y, dtype=tf.float32)
        # Define the weights and computation.
        w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = w * x_data + b
        # Define the loss.
        self.loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        self.grads = optimizer.compute_gradients(self.loss)
        self.train = optimizer.apply_gradients(self.grads)
        # Define the weight initializer and session.
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        # Additional code for setting and getting the weights
        self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
        # Return all of the data needed to use the network.
        self.sess.run(init)

    # Define a remote function that trains the network for one step and returns the
    # new weights.
    def step(self, weights):
        # Set the weights in the network.
        self.variables.set_weights(weights)
        # Do one step of training. We only need the actual gradients so we filter over the list.
        actual_grads = self.sess.run([grad[0] for grad in self.grads])
        return actual_grads

    def get_weights(self):
        return self.variables.get_weights()

# Define a remote function for generating fake data.
@ray.remote(num_return_vals=2)
def generate_fake_x_y_data(num_data, seed=0):
    # Seed numpy to make the script deterministic.
    np.random.seed(seed)
    x = np.random.rand(num_data)
    y = x * 0.1 + 0.3
    return x, y

# Generate some training data.
batch_ids = [generate_fake_x_y_data.remote(BATCH_SIZE, seed=i) for i in range(NUM_BATCHES)]
x_ids = [x_id for x_id, y_id in batch_ids]
y_ids = [y_id for x_id, y_id in batch_ids]
# Generate some test data.
x_test, y_test = ray.get(generate_fake_x_y_data.remote(BATCH_SIZE, seed=NUM_BATCHES))

# Create actors to store the networks.
remote_network = ray.remote(Network)
actor_list = [remote_network.remote(x_ids[i], y_ids[i]) for i in range(NUM_BATCHES)]
local_network = Network(x_test, y_test)		# 此处和上边的多了一个本地实例化类

# Get initial weights of local network.
weights = local_network.get_weights()

# Do some steps of training.
for iteration in range(NUM_ITERS):
    # Put the weights in the object store. This is optional. We could instead pass
    # the variable weights directly into step.remote, in which case it would be
    # placed in the object store under the hood. However, in that case multiple
    # copies of the weights would be put in the object store, so this approach is
    # more efficient.
    weights_id = ray.put(weights)
    # Call the remote function multiple times in parallel.
    gradients_ids = [actor.step.remote(weights_id) for actor in actor_list]
    # Get all of the weights.
    gradients_list = ray.get(gradients_ids)

    # Take the mean of the different gradients. Each element of gradients_list is a list
    # of gradients, and we want to take the mean of each one.
    mean_grads = [sum([gradients[i] for gradients in gradients_list]) / len(gradients_list) for i in range(len(gradients_list[0]))]

    feed_dict = {grad[0]: mean_grad for (grad, mean_grad) in zip(local_network.grads, mean_grads)}
    local_network.sess.run(local_network.train, feed_dict=feed_dict)
    weights = local_network.get_weights()

    # Print the current weights. They should converge to roughly to the values 0.1
    # and 0.3 used in generate_fake_x_y_data.
    if iteration % 20 == 0:
        print("Iteration {}: weights are {}".format(iteration, weights))
```



## 3. 故障分析

请注意，`TensorFlowVariables`使用变量名来确定调用`set_weights`时要设置的变量。 当在同一TensorFlow图中定义两个网络时，会出现一个常见问题。 在这种情况下，TensorFlow将下划线和整数附加到变量名称以消除它们的歧义。 这将导致`TensorFlowVariables`失败。 例如，如果我们定义一个带有`TensorFlowVariables`实例的网络`Network`：

```
import ray
import tensorflow as tf

class Network(object):
    def __init__(self):
        a = tf.Variable(1)
        b = tf.Variable(1)
        c = tf.add(a, b)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        self.variables = ray.experimental.tf_utils.TensorFlowVariables(c, sess)

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def get_weights(self):
        return self.variables.get_weights()
```

继续运行下面代码：

```
a = Network()
b = Network()
b.set_weights(a.get_weights())
```
代码会报错。如果我们将每个网络`Network`定义在它自己的张量流图中，那么它就会工作:

```
with tf.Graph().as_default():
    a = Network()
with tf.Graph().as_default():
    b = Network()
b.set_weights(a.get_weights())
```
在包含network的actor之间不会发生此问题，因为每个actor都在其自己的进程中，因此在其自己的图中。 使用`set_flat`时也不会发生这种情况。
要记住的另一个问题是`TensorFlowVariables`需要向图计算中添加新操作。 如果关闭图并使其不可变，例如 创建`MonitoredTrainingSession`初始化将失败。 要解决此问题，只需在关闭图之前创建实例。

## 4.TensorFlowVariables API
| class ray.experimental.tf_utils.TensorFlowVariables(output, sess=None, input_variables=None) |  
|:--|--|
源码文末附录。

用于为Tensorflow 网络设置和获取权重的类。

**sess**
用于运行赋值的tensorflow会话。
*Type:	tf.Session*

**variables**
从传入的loss或附加变量中提取变量。
*Type:	Dict[str, tf.Variable]*

**placeholders**
占位符权重。
*Type:	Dict[str, tf.placeholders]*

**assignment_nodes**
分配权重的节点。
*Type:	Dict[str, tf.Tensor]*

**set_session(sess)**
设置类当前使用的会话。
参数：	sess (tf.Session) –会话

**get_flat_size()**
返回所有扁平变量的总长度。

**get_flat()**
获取权重并以flat 数组的形式返回。
*返回：包含压扁权值的一维数组。*

**set_flat(new_weights)**
将权重设置为new_weights，从flat 数组转换而来。
参数：new_weights (np.ndarray) –包含权重的flat数组。
**注：** 只能使用此函数设置网络中的所有权重，即，数组的长度必须与get_flat_size匹配。

**get_weights()**
返回一个包含网络权重的字典。

**set_weights(new_weights)**
将权重设置为new_weights。
参数：new_weights (Dict) –字典将变量名映射到它们的权重。
**注：** 也可以设置变量的子集，只需传入需要设置的变量。
## 附录
**ray.experimental.tf_utils**

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque, OrderedDict
import numpy as np

import tensorflow as tf


def unflatten(vector, shapes):
    i = 0
    arrays = []
    for shape in shapes:
        size = np.prod(shape, dtype=np.int)
        array = vector[i:(i + size)].reshape(shape)
        arrays.append(array)
        i += size
    assert len(vector) == i, "Passed weight does not have the correct shape."
    return arrays


[docs]class TensorFlowVariables(object):
    """A class used to set and get weights for Tensorflow networks.

    Attributes:
        sess (tf.Session): The tensorflow session used to run assignment.
        variables (Dict[str, tf.Variable]): Extracted variables from the loss
            or additional variables that are passed in.
        placeholders (Dict[str, tf.placeholders]): Placeholders for weights.
        assignment_nodes (Dict[str, tf.Tensor]): Nodes that assign weights.
    """

    def __init__(self, output, sess=None, input_variables=None):
        """Creates TensorFlowVariables containing extracted variables.

        The variables are extracted by performing a BFS search on the
        dependency graph with loss as the root node. After the tree is
        traversed and those variables are collected, we append input_variables
        to the collected variables. For each variable in the list, the
        variable has a placeholder and assignment operation created for it.

        Args:
            output (tf.Operation, List[tf.Operation]): The tensorflow
                operation to extract all variables from.
            sess (tf.Session): Session used for running the get and set
                methods.
            input_variables (List[tf.Variables]): Variables to include in the
                list.
        """
        self.sess = sess
        if not isinstance(output, (list, tuple)):
            output = [output]
        queue = deque(output)
        variable_names = []
        explored_inputs = set(output)

        # We do a BFS on the dependency graph of the input function to find
        # the variables.
        while len(queue) != 0:
            tf_obj = queue.popleft()
            if tf_obj is None:
                continue
            # The object put into the queue is not necessarily an operation,
            # so we want the op attribute to get the operation underlying the
            # object. Only operations contain the inputs that we can explore.
            if hasattr(tf_obj, "op"):
                tf_obj = tf_obj.op
            for input_op in tf_obj.inputs:
                if input_op not in explored_inputs:
                    queue.append(input_op)
                    explored_inputs.add(input_op)
            # Tensorflow control inputs can be circular, so we keep track of
            # explored operations.
            for control in tf_obj.control_inputs:
                if control not in explored_inputs:
                    queue.append(control)
                    explored_inputs.add(control)
            if ("Variable" in tf_obj.node_def.op
                    or "VarHandle" in tf_obj.node_def.op):
                variable_names.append(tf_obj.node_def.name)
        self.variables = OrderedDict()
        variable_list = [
            v for v in tf.global_variables()
            if v.op.node_def.name in variable_names
        ]
        if input_variables is not None:
            variable_list += input_variables
        for v in variable_list:
            self.variables[v.op.node_def.name] = v

        self.placeholders = {}
        self.assignment_nodes = {}

        # Create new placeholders to put in custom weights.
        for k, var in self.variables.items():
            self.placeholders[k] = tf.placeholder(
                var.value().dtype,
                var.get_shape().as_list(),
                name="Placeholder_" + k)
            self.assignment_nodes[k] = var.assign(self.placeholders[k])

[docs]    def set_session(self, sess):
        """Sets the current session used by the class.

        Args:
            sess (tf.Session): Session to set the attribute with.
        """
        self.sess = sess


[docs]    def get_flat_size(self):
        """Returns the total length of all of the flattened variables.

        Returns:
            The length of all flattened variables concatenated.
        """
        return sum(
            np.prod(v.get_shape().as_list()) for v in self.variables.values())


    def _check_sess(self):
        """Checks if the session is set, and if not throw an error message."""
        assert self.sess is not None, ("The session is not set. Set the "
                                       "session either by passing it into the "
                                       "TensorFlowVariables constructor or by "
                                       "calling set_session(sess).")

[docs]    def get_flat(self):
        """Gets the weights and returns them as a flat array.

        Returns:
            1D Array containing the flattened weights.
        """
        self._check_sess()
        return np.concatenate([
            v.eval(session=self.sess).flatten()
            for v in self.variables.values()
        ])


[docs]    def set_flat(self, new_weights):
        """Sets the weights to new_weights, converting from a flat array.

        Note:
            You can only set all weights in the network using this function,
            i.e., the length of the array must match get_flat_size.

        Args:
            new_weights (np.ndarray): Flat array containing weights.
        """
        self._check_sess()
        shapes = [v.get_shape().as_list() for v in self.variables.values()]
        arrays = unflatten(new_weights, shapes)
        placeholders = [
            self.placeholders[k] for k, v in self.variables.items()
        ]
        self.sess.run(
            list(self.assignment_nodes.values()),
            feed_dict=dict(zip(placeholders, arrays)))


[docs]    def get_weights(self):
        """Returns a dictionary containing the weights of the network.

        Returns:
            Dictionary mapping variable names to their weights.
        """
        self._check_sess()
        return {
            k: v.eval(session=self.sess)
            for k, v in self.variables.items()
        }


[docs]    def set_weights(self, new_weights):
        """Sets the weights to new_weights.

        Note:
            Can set subsets of variables as well, by only passing in the
            variables you want to be set.

        Args:
            new_weights (Dict): Dictionary mapping variable names to their
                weights.
        """
        self._check_sess()
        assign_list = [
            self.assignment_nodes[name] for name in new_weights.keys()
            if name in self.assignment_nodes
        ]
        assert assign_list, ("No variables in the input matched those in the "
                             "network. Possible cause: Two networks were "
                             "defined in the same TensorFlow graph. To fix "
                             "this, place each network definition in its own "
                             "tf.Graph.")
        self.sess.run(
            assign_list,
            feed_dict={
                self.placeholders[name]: value
                for (name, value) in new_weights.items()
                if name in self.placeholders
            })

```

