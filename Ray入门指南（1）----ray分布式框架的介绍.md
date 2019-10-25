

使用ray，你需要解决以下两个问题：
1.ray执行异步任务时是怎样实现平行的。（ray集群中每个节点共享本节点本地存储，节点中worker并行运行，集群间worker的并行）
2.ray是怎样使用对象ID来表示不可变对象的远程对象的。（任务执行前，对象值已存入存储对象中，任务执行是通过对象ID调用存储对象的）

## **综述**
Ray是一个分布式执行引擎。同样的代码可以再一台机器上实现高效的多处理，也可以在集群是用于大型的计算。

当我们使用Ray时，涉及到多个进程。

 - **多worker进程执行多个任务并将结果存储在对象存储中，每个worker都是一个独立的进程。**
 - **每个节点上的对象存储都将不可变的对象存储存储在共享内存中，允许worker以少量的复制和并行化有效的共享同一节点上的存储对象。**
 - **每一节点上的本地调度将任务分配给同一节点上的worker（一个节点上的本地调度把任分配给本节点的worker）**
 - **一个driver是用户控制的python进程。例如，如果用户正在运行脚本或者使用python shell，那么driver就是运行脚本或者shell的python进程。driver和worker很相似，他们都可以提交任务给本地调度并从对象存储中获取对象，但是不同之处是本地调度不会将任务分配给driver执行。**
 - **Redis服务器维护系统的大部分状态。例如，它跟踪哪些对象位于哪些机器上，以及任务规范(但不包括数据)。另外可以对有问题的目标进行直接的质问（就是提示错误）。**

## **运行Ray**
启动python并运行一下命令：

    import ray
    ray.init()
ray启动了。（其中ray.init()方法可以加参数，具体请参考Ray相关的[API](https://blog.csdn.net/weixin_43255962/article/details/88850456)）

## 不可变的远程(remote)对象
在Ray中，我们可以在对象上创建和计算。我们将这些对象称为远程（remote）对象，并使用对象ID来引用它们。remote对象是被存储在对象存储中的，在集群中每个节点都有一个存储对象。在集群设置中，我们可能实际上不知道每个存储对象的位置。
对象ID本质上是一个唯一的ID，可以被用作引用远程（remote）对象。如果您熟悉future（期货），它和对象ID是很相似的。
**我们规定远程对象是不可变的。也就是说，它们的值在创建之后不能更改。这允许在多个对象存储中复制远程对象，而不需要同步副本。**

## Put和Get
`ray.get`和`ray.put`是用作python对象和对象ID之间的转换，下边是一个例子。

    x = "example"
    ray.put(x)  # ObjectID(b49a32d72057bdcfc4dda35584b3d838aad89f5d)
ray.put(x)命令的运行是通过worker进程或者driver进程（dervier进程是正在运行的脚本）。它把一个python对象复制到本地对象存储中(这里的本地意味着在同一节点上)。**一旦对象的被存入存储对象后，他的值就不能被改变了。**
此外，`ray.put(x)` **返回的是一个对象ID，它本质上是一个ID，可被引用新创建的远程（remote）对象**。如果我们保存一个对象ID在一个变量（`x_id=ray.put(x)`）中,然后我们可以将`x_id`传递给远程函数，这些远程函数将对相应的远程对象进行操作。
命令`ray.get(x_id)`接受一个对象ID，并从相应的远程对象创建一个Python对象。对于像数组这样的对象，我们可以使用内存的共享从而避免复制对象。对于其他对象，它将对象从对象存储中复制到worker进程的堆。如果与对象ID `x_id`对应的远程（remote）对象与调用`ray.get(x_id)`的worker不在同一个节点上，那么远程（remote）对象将首先从拥有它的对象存储区转移到需要它的对象存储区。

    x_id = ray.put("example")
    ray.get(x_id)  # "example"
如果与对象ID `x_id`对应的远程对象尚未创建，则命令`ray.get(x_id)`将等待创建远程对象。
ray的一个非常常见的用例`ray.get`是获取对象id列表。在本例中，您可以调用`ray.get(object_id)`，其中`object_id`是对象id的列表。

    result_ids = [ray.put(i) for i in range(10)]
    ray.get(result_ids)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ray.get(result_ids[5]) 	# 5
    ray.get(result_ids[1]) 	# 1

## **Ray的异步计算**
Ray支持异步执行任意Python函数。这是通过将Python函数指定为**远程函数**来实现的。

例如，一个普通的Python函数是这样的。

    def add1(a, b):
        return a + b
远程函数是这样的。

    @ray.remote
    def add2(a, b):
        return a + b

## 远程函数（remote functions）
调用`add1(1,2`)返回`3`，并导致Python解释器阻塞，直到计算完成，然而调用`add2.remote(1,2)`立即返回一个对象ID并创建一个任务。该任务将由系统调度并异步执行(可能在不同的机器上)。当任务完成执行时，它的返回值将存储在对象存储中。

    x_id = add2.remote(1, 2)
    ray.get(x_id)  # 3
下面的简单示例演示如何使用异步任务并行化计算。

    import time
    
    def f1():
        time.sleep(1)
    
    @ray.remote
    def f2():
        time.sleep(1)
    
    # 这个操作需要10秒.
    [f1() for _ in range(10)]
    
    # 下面的操作只需要一秒钟(假设系统至少有10个cpu)
    ray.get([f2.remote() for _ in range(10)])
**提交任务和执行任务之间有明显的区别。当调用远程函数时，将执行该函数的任务提交给本地调度程序，并立即返回任务输出的对象ID。但是，直到系统将任务实际调度到worker上，任务才会执行。**　任务执行不是懒洋洋地完成的。系统将输入数据移动到任务中，只要任务的输入依赖项可用，并且有足够的资源进行计算，任务就会立马执行。
**当提交任务时，每个参数可以通过值或对象ID传入。**　简单的字符数字会以值直接传入，远程对象或者数组等对象会以对象ID传入。例如，这些行具有相同的行为。

    add2.remote(1, 2)
    add2.remote(1, ray.put(2))
    add2.remote(ray.put(1), ray.put(2))
远程函数不返回实际值，它们总是返回对象ID。
远程对象的执行实际上是对python对象的操作。也就是说，如果使用任何对象id调用远程函数，系统将从对象存储中检索相应的对象。
注意，一个远程函数可以返回多个对象ID。

    @ray.remote(num_return_vals=3)
        def return_multiple():
            return 1, 2, 3
         a_id, b_id, c_id = return_multiple.remote()

## 任务之间的依赖关系
程序员可以通过将一个任务的对象ID输出作为参数传递给另一个任务来表达任务之间的依赖关系。例如，我们可以启动以下三个任务，每个任务都依赖于前一个任务。

    @ray.remote
    def f(x):
        return x + 1
    
    x = f.remote(0)
    y = f.remote(x)
    z = f.remote(y)
    ray.get(z) # 3
上面的第二个任务在第一个任务完成之前不会执行，第三个任务在第二个任务完成之前不会执行。在这个例子中，没有体现并行。
组合的任务更能体现任务之间的依赖关系。考虑以下树缩减的实现。

```
import numpy as np

@ray.remote
def generate_data():
    return np.random.normal(size=1000)

@ray.remote
def aggregate_data(x, y):
    return x + y

# 生成一些随机数据。这将启动100个任务，这些任务将在各个节点上调度。
# 结果数据将分布在集群的各个节点中。此时date的ID内存中有100*1000个数据
data = [generate_data.remote() for _ in range(100)]

# 执行树缩减。
while len(data) > 1:
    data.append(aggregate_data.remote(data.pop(0), data.pop(0)))

#获取结果 1000个数据
ray.get(data)
```

## 远程函数中的远程函数
到目前为止，我们只从driver进程调用远程函数。但是worker进程也可以调用远程函数。为了说明这一点，请考虑下面的示例。

```
@ray.remote
def sub_experiment(i, j):
    # Run the jth sub-experiment for the ith experiment.
    return i + j

@ray.remote
def run_experiment(i):
    sub_results = []
    # Launch tasks to perform 10 sub-experiments in parallel.
    for j in range(10):
        sub_results.append(sub_experiment.remote(i, j))
    # Return the sum of the results of the sub-experiments.
    return sum(ray.get(sub_results))

results = [run_experiment.remote(i) for i in range(5)]
ray.get(results) # [45, 55, 65, 75, 85]
```
当远程函数`run_experiment`在一个worker上执行时，它会多次调用远程函数`sub_experiment`。这个例子，说明了如何并行地运行多个实验，每个实验都利用了内部的并行性。

此篇主要参考Ray官网，如有错误，请阅读者提出指正，谢谢！
原英文链接：https://ray.readthedocs.io/en/latest/tutorial.html  
ray API 中文：https://blog.csdn.net/weixin_43255962/article/details/88850456
