本篇主要介绍Ray在集群中的使用（启动及停止）并通过一个例子测试Ray是否启动成功（主要适应小型集群。对于较大集群可以考虑使用pass包：`sudo apt-get install pssh`或[私有集群的安装命令](https://ray.readthedocs.io/en/latest/autoscaling.html#quick-start-private-cluster)。）

## 在集群上部署Ray

假设有一个正在运行的集群，并且集群中的节点可以彼此通信，另外还假定Ray安装在每台机器上。
#### 在每台机器上启动Ray
**１．在head节点上(只需选择某个节点作为head节点)运行以下操作**。如果省略了`--redis-port`参数，Ray将随机选择一个端口。

```
ray start --head --redis-port=6379
```
该命令将打印出启动的Redis服务器的地址(以及一些其他地址信息)。
**２．然后在所有其他节点上运行以下操作。**确保将`<redis-address>`替换为head节点上命令打印的值(例如：`123.45.67.89:6379`)。

```
ray start --redis-address=<redis-address>   
```
所有的子节点上运行的启动命令中的地址是相同的，即`--redis-address=<redis-address>`  是相同的 。

如果您希望指定一台机器有10个cpu和1个GPU，您可以使用标志`--numcpu =10`和`--numgpu =1`来实现这一点。如果不使用这些标志，那么Ray将自动检测cpu的数量，并假设有0个gpu。

现在我们已经启动了每个节点上的所有Ray进程。这包括

 - 每台机器上有一些worker进程。
 - 每天机器上有一个对象存储。
 - 每台机器上有一个raylet。
 - 多个Redis服务器(位于head节点上)。
 ##### 　运行的python代码例子
 当主节点及子节点都被启动后，可以运行一下代码进行测试ray是否已经成功启动并连通。

下边代码是初始化ray。其中`ray.init()`,初始化ray，即链接到本地安装的ray，如果想连接到启动的ray集群上需要通过加上参数`redis_address="<redis-address>"`，其中的`<redis-address>`是主节点的ip地址和端口号，例如：`ray.init(redis_address="10.10.10.133:6379")`，主节点的ip地址是`10.10.10.133`,端口号是`6379`。
```
import ray
ray.init(redis_address="<redis-address>")
```
现在可以定义远程函数并执行任务。例如，要验证加入集群的节点数量是否正确，可以运行以下命令。

```
import time

@ray.remote
def f():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()	# 返回本节点对应的ip地址
	
# Get a list of the IP addresses of the nodes that have joined the cluster.
set(ray.get([f.remote() for _ in range(1000)]))		#　set()是剔除重复项
```

代码运行1000次是为，把任务尽量都分发到各个节点上，进而饭后所有节点对象的ip地址。因为一个节点上可能同时（平行）执行多个任务。


#### 停止Ray
当您想要停止Ray进程时，请在每个节点上运行`ray stop`。

**注意**
１．启动ray时需要线启动主节点，再启动从节点（子节点）。主节点上运行`ray start --head　--redis-port=6379`,从节点运行运行`ray start --redis-address=<redis-address>`   ,`redis-address`包括主节点ip和端口号。
２．停止ray的时，先停止从节点，再停止主节点。特别是正在运行程序的时候，直接停止主节点，可能会造成从节点上python异常（丢失python模块）。
