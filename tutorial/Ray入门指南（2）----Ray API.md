## Ray API
此篇主要介绍Ray的API。主要有两部分，一部分是ray在python代码中的应用；另一部分是ray在命令行中的应用。
```
ray.init(redis_address=None, num_cpus=None, num_gpus=None, resources=None,
		 object_store_memory=None, redis_max_memory=None, log_to_driver=True, 
		 node_ip_address=None, object_id_seed=None, local_mode=False, 
		 redirect_worker_output=None, redirect_output=None, ignore_reinit_error=False,
		 num_redis_shards=None, redis_max_clients=None, redis_password=None, 
		 plasma_directory=None, huge_pages=False, include_webui=False, 
		 driver_id=None, configure_logging=True, logging_level=20, 
		 logging_format='%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s', 
		 plasma_store_socket_name=None, raylet_socket_name=None, temp_dir=None,
		  load_code_from_local=False, _internal_config=None)
```
连接到现有的Ray集群，或者启动一个Ray集群(单个电脑可以理解为１个节点的集群)。

此方法处理两种情况。要么一个Ray集群已经存在，我们将这个驱动程序附加到它上面，要么启动所有与Ray集群相关的进程并附加到新启动的集群上。

要启动Ray和所有相关进程，请使用以下步骤:

    ray.init()
要连接到现有的Ray集群，可以使用以下方法(替换为适当的地址):

    ray.init(redis_address="123.45.67.89:6379")
```
**参数:**	
redis_address (str) – 要连接到的Redis服务器的地址。如果不提供此地址，则此命令将启动Redis、
					a global scheduler, a local scheduler, a plasma store, a plasma manager, 和 
					some workers.它还将在Python退出时终止这些进程。
num_cpus (int) – 用户希望配置所有本地调度程序的cpu数量，如果没有此选项则默认是调用本地的所有cpu数。
num_gpus (int) – 用户希望配置所有本地调度程序的gpu数量，如果没有此选项则默认是调用本地的所有gpu数。
resources – 将资源名称映射到可用资源数量的字典。（就是将资源映射到redis中）
object_store_memory – 用于启动对象存储的内存量(以字节为单位)。默认情况下，这个上限为20GB，但可以设置得更高。
redis_max_memory – 允许每个redis碎片使用的最大内存量(以字节为单位)。一旦超过这个限制，redis将开始LRU清除条目。
					这只适用于切分的redis表(任务、对象和概要表)。默认情况下，这是10GB的，但可以设置得更高。
log_to_driver (bool) – 如果为真，则所有节点上的所有工作进程的输出都将定向到driver。
node_ip_address (str) – 当前所在节点的IP地址。
object_id_seed (int) – 用于确定对象ID的生成。可以在同一作业的多个运行中使用相同的值，以便以一致的方式生成对象ID。
						但是，不同的作业不应该使用相同的ID。
local_mode (bool) – 如果代码在没有Ray的情况下连续执行，则为True。这对于调试非常有用。
ignore_reinit_error – 如果我们应该禁止第二次调用ray.init()时出现错误，则为True。
num_redis_shards – 除了主Redis碎片之外，还要启动的Redis碎片的数量。
redis_max_clients –设置最大客服端数量。
redis_password (str) – 防止没有密码的外部客户端连接到Redis(如果提供)。
plasma_directory – 将创建plasma内存映射文件的目录
huge_pages – 布尔标志，指示是否使用hugetlbfs支持启动对象存储。需要plasma_directory。
include_webui – 布尔标志，指示是否启动web UI，它显示Ray集群的状态。
driver_id –  driver的ID.
configure_logging – 如果允许这里的日志编码，则为真。否则，用户可能希望自己配置它。
logging_level –日志级别，默认为logging.info。.
logging_format – 日志格式，默认包含时间戳、文件名、行号和消息。看到ray_constants.py。
plasma_store_socket_name (str) – 如果提供，它将指定plasma存储使用的套接字名称。
raylet_socket_name (str) –如果提供，它将指定raylet进程使用的套接字路径。
temp_dir (str) –如果提供，它将为Ray进程指定根临时目录。
load_code_from_local – 是否应该从本地模块或从GCS加载代码。
_internal_config (str) – 用于覆盖RayConfig默认值的JSON配置。仅供测试之用。
```

```
Returns:	处理有关已启动进程的信息。
```

```
Raises:	Exception –如果传入不适当的参数组合，则会引发异常。
```
**ray.is_initialized()**
检查ray.init 是否已经调用（即检查ray是否启动）。
返回：如果已经调用ray.init 返回true ，否则返回false。

ray.remote(*args, **kwargs)
定义一个远程函数或一个actor类。
这可以在没有参数的情况下定义远程函数或角色（actor），如下所示:
```
@ray.remote
def f():
    return 1

@ray.remote
class Foo(object):
    def method(self):
        return 1
```
它也可以用于特定的关键字参数:
*num_return_vals:*  这只适用于远程函数。它指定远程函数调用返回的对象id的数量。
*num_cpus:*   为该任务或actor的生命周期保留的CPU内核数量。
*num_gpus:*   为该任务或actor的生命周期保留的GPU数量。
*resources:*  为此任务或actor的生命周期保留的各种自定义资源的数量。这是一个将字符串(资源名称)映射到数字的字典。
*max_calls:* 仅适用于远程功能。这指定了一个给定的worker在必须退出之前可以执行给定的远程函数的最大次数(这可以用于解决第三方库中的内存泄漏，或者回收不能轻松释放的资源，例如TensorFlow获得的GPU内存)。默认情况下这是无限的。（获取资源的次数）
*max_reconstructions:* 这指定当actor意外死亡时，应该重构它的最大次数。最小有效值为0(默认值)，这表示不需要重构参与者。最大有效值是ray.ray_constants. .INFINITE_RECONSTRUCTIONS。
例:

```
@ray.remote(num_gpus=1, max_calls=1, num_return_vals=2)
def f():
    return 1, 2

@ray.remote(num_cpus=2, resources={"CustomResource": 1})
class Foo(object):
    def method(self):
        return 1
```
**ray.get(object_ids)**
从对象存储中获得远程对象或远程对象列表。
这是一个阻塞方法，直到对象ID对应的对象在本地对象存储可用为止。如果这个对象不在本地对象存储中，那么它将从拥有它的对象存储中发出(创建了此对象的对象存储)。如果object_ids是一个列表，那么将返回与列表中每个对象对应的对象。

    Parameters（参数）:object_ids 要获取的对象的对象ID或要获取的对象ID列表。
    Returns（返回）:一个Python对象或一组Python对象。

**ray.wait(object_ids, num_returns=1, timeout=None)**
返回一个准备好的ID列表和一个没有准备的ID列表（暂时没有创建的id）。

  

 - 超时参数过去以毫秒为单位，现在以秒为单位。

如果设置了timeout，则函数将在准备好请求的id数量或达到超时时返回，无论先发生什么情况。如果没有设置该函数，则该函数只是等待，直到对象数量准备好并返回确切数量的对象id。
这个方法返回两个列表。第一个列表由对象id组成，这些id对应于对象存储中可用的对象。第二个列表对应于其余的对象id(可能已经准备好了，也可能还没有准备好)。
对象id的输入列表的顺序保持不变。也就是说，如果A在输入列表中位于B之前，并且两者都在就绪列表中，那么A将在就绪列表中位于B之前。如果A和B都在剩下的列表中，这也是成立的。

    Parameters（参数）:object_ids (List[ObjectID]) -–对象的对象id列表，这些对象可能已经准备好，也可能还没有准备好。
    												注意，这些id必须是惟一的。
    				num_returns(int)--应该返回的对象ID 数
    				timeout (float) -- 返回前等待的最大时间(以秒为单位)。
    Returns（返回）:num_returns (int) --已准备好的对象id列表和剩余对象id列表。
**ray.put(value)**

    把对象存储在存储对象中。
    Parameters（参数）:	value –要存储的Python对象。
    Returns（返回）:分配给此值的对象ID。

**ray.get_gpu_ids()**
获取worker中可用的gpu的IDs。
如果在worker启动时设置了CUDA_VISIBLE_DEVICES环境变量，那么该方法返回的id将是CUDA_VISIBLE_DEVICES中id的子集。否则，id将位于[0,num_gpu - 1]范围内，其中num_gpu是节点拥有的gpu数量。

    Returns（返回）:	 GPU的 ID列表.
**ray.get_resource_ids()**
获取worker可用资源的id。

    Returns（返回）:将资源的名称映射到对列表的字典，其中每对由资源的ID和为该worker保留的资源的一部分组成。

**ray.get_webui_url()**
获取访问web UI的URL（路径）。
注意，URL没有指定web UI所在的节点

    Returns（返回）:	web UI的URL的字符串。
**ray.shutdown(exiting_interpreter=False)**
断开worker的连接，并终止由ray.init()启动的进程。
当使用Ray的Python进程退出时，这将在末尾自动运行。可以连续运行两次。该函数的主要用例是清除测试之间的状态。
注意，这将清除任何远程函数定义、actor定义和现有actor，因此，如果希望在调用ray.shutdown()之后使用任何以前定义的远程函数或actor，那么需要重新定义它们。如果它们是在导入模块中定义的，则需要重新加载模块。
Parameters:	exiting_interpreter (bool) –如果是由退出程序来执行则为true。如果我们退出解释器，我们将等待一段时间来打印任何额外的错误消息。
**ray.register_custom_serializer(cls, use_pickle=False, use_dict=False, serializer=None, deserializer=None, local=False, driver_id=None, class_id=None)**
为特定类启用序列化和反序列化。
这个方法在每个worker上运行下面定义的register_class函数，这将使ray能够正确地序列化和反序列化该类的对象。

    Parameters（参数）:
    		cls (type) –ray应该使用这个自定义序列化器的类。 
    		use_pickle (bool)--如果为真，则使用pickle序列化该类的对象。
    		use_dict –如果为真，则序列化该类的对象，将它们的_dict__字段转换为字典。如果use_pickle为真，则必须为假。
    		serializer – 要使用的自定义序列化程序。当且仅当use_pickle和use_dict为False时，才应该提供此选项。
    		deserializer –要使用的自定义反序列化器。当且仅当use_pickle和use_dict为False时，才应该提供此选项。
    		local –如果序列化器只应在当前worker上注册，则为真。这通常是假。
    		driver_id –要注册该类的驱动程序（driver）的ID。
    		class_id –我们正在注册的类的ID。如果没有指定，我们将在函数中计算一个新的。
    
    Raises:
    		Exception – 如果pickle=False且类不能被Ray有效序列化，则会引发异常。
    					如果use_dict为真且cls不可选，这也会引发异常。
**ray.profile(event_type, extra_data=None)**
分析时间跨度，使其出现在时间轴可视化中。
注意，这只在raylet代码路径中有效。
该函数可以如下方式使用(在驱动程序上或任务中)。

```
with ray.profile("custom event", extra_data={'key': 'value'}):
    # Do some computation here.
```
另外，字典可以作为“extra_data”参数传递，如果您想覆盖默认的时间轴显示文本和框颜色，它可以有键“name”和“cname”。当您单击与此配置文件跨度对应的框时，其他值将出现在chrome跟踪GUI的底部。

    Parameters:
    			event_type –描述事件类型的字符串。
    			extra_data –这必须是一个将字符串映射到字符串的字典。这些数据将被添加到用于填充时间轴的json对象中，
    						因此，如果想设置特定的颜色，只需将“cname”属性设置为适当的颜色。
    						类似地，如果您设置了“name”属性，那么它将设置时间轴上框中显示的文本。
    Returns:可以通过“with”语句分析时间跨度的对象。
**ray.method(args, kwargs)**
注释一个actor方法。
例：

```
@ray.remote
class Foo(object):
    @ray.method(num_return_vals=2)
    def bar(self):
        return 1, 2

f = Foo.remote()

_, _ = f.bar.remote()
```

    Parameters:num_return_vals – 调用此actor方法应返回的对象id的数量。

**

## Ray命令行API

**

**ray start（ray启动）**

    ray start [OPTIONS]

|  非官方个   |  
| --- | --- | 
|     |     |   
|     |     |   
|     |     |   

*Options*

| --node-ip-address <node_ip_address> | 　 
| --- | --- |
|此节点的IP地址  | 　 

| --redis-address <redis_address> |  
|--|--|
|用于连接到Redis的地址  |  |

|--redis-port <redis_port>|  
|--|--|
|用于启动Redis的端口 |  

|--num-redis-shards <num_redis_shards>  |  
|--|--|
|  除主Redis碎片外，要使用的其他Redis碎片的数量|  

|--redis-max-clients <redis_max_clients>|  
|--|--|
| 如果提供，请尝试使用此最大客户端数量配置Redis。 |  |

|--redis-password <redis_password>|  
|--|--|
|如果提供了该密码，则使用该密码进行安全的Redis端口 |  |

|--redis-shard-ports <redis_shard_ports>|  
|--|--|
| 用于除主Redis碎片之外的Redis碎片的端口 |  |

|  --object-manager-port <object_manager_port>|  
|--|--|
|  用于启动对象管理器的端口|  |

|  --object-store-memory <object_store_memory>  |
|--|--|
| 用于启动对象存储的内存量(以字节为单位)。默认情况下，这个上限为20GB，但可以设置得更高。 |  |

|--redis-max-memory <redis_max_memory>  |
|--|--|
|  允许redis使用的最大内存量(以字节为单位)。一旦超过这个限制，redis将开始LRU清除条目。这只适用于切分的redis表(任务、对象和概要表)。默认情况下，这是10GB的上限，但可以设置得更高。|  |

|--num-cpus <num_cpus>  |
|--|--|
| 此节点上的cpu数量 |  |

|--num-gpus <num_gpus>  |
|--|--|
|  此节点上的gpu数量|  |

|--resources <resources>  |
|--|--|
|  将资源名称映射到资源数量的JSON序列化字典|  |

|--head  |
|--|--|
|为head节点提供这个参数  |  |

|--include-webui|  
|--|--|
| 如果应该启动UI，则提供此参数 |  |

|--block|  
|--|--|
|  在此命令中提供此参数以永久阻塞|  |

|--plasma-directory <plasma_directory>|  
|--|--|
|对象存储用于内存映射文件的目录  |  |

|--huge-pages| 
|--|--|
|  在对象存储中启用对大页面的支持|  |

|--autoscaling-config <autoscaling_config>  |
|--|--|
|  包含自动缩放配置的文件|  |

|--no-redirect-worker-output  |
|--|--|
|不重定向worker的标准输出和标准derr到文件  |  |

|  --plasma-store-socket-name <plasma_store_socket_name>  |
|--|--|
| 手动指定等plasma存储的套接字名称 |  |

|--raylet-socket-name <raylet_socket_name> |
|--|--|
|  手动指定raylet进程的套接字路径|  |

|--temp-dir <temp_dir>  |
|--|--|
|  手动指定Ray进程的根临时目录|  |

|--include-java  |
|--|--|
| 启用Java worker支持。 | |

|--java-worker-options <java_worker_options>  |
|--|--|
|覆盖启动Java worker的选项。  |  |

|--internal-config <internal_config>  |
|--|--|
|不要使用这个。这仅用于调试/开发目的。  |  |

|--load-code-from-local  |
|--|--|
| 指定从本地文件加载代码还是从GCS序列化加载代码。 |  |

**ray stop（ray 关闭）**

    ray stop [OPTIONS]

**ray up**
创建或更新一个Ray集群。

    ray up [OPTIONS] CLUSTER_CONFIG_FILE
*Options*
|--no-restart   |
|--|--|
| 是否在更新期间跳过重新启动Ray服务。这样可以避免中断正在运行的作业。  |

|  --restart-only|
|--|--|
|  是否跳过运行设置命令，只重启Ray。这不能与“no-restart”一起使用。|  |

|--min-workers <min_workers>  |
|--|--|
| 覆盖已配置的集群最小worker节点计数 |  |

|--max-workers <max_workers>  |
|--|--|
|  覆盖已配置的集群最大worker节点计数|  |

|-n, --cluster-name <cluster_name>  |
|--|--|
| 覆盖已配置的集群名称。 |  |

|-y, --yes  |
|--|--|
|不要要求确认。  |  |

*Arguments*

|CLUSTER_CONFIG_FILE  |
|--|--|
| 需要的参数 |  |

**ray down**
拆下Ray集群部件

```
ray down [OPTIONS] CLUSTER_CONFIG_FILE
```
*Options*

|--workers-only  |
|--|--|
|只摧毁worker。  |  |

|-y, --yes  |
|--|--|
| 不要求确认 |  |

|-n, --cluster-name <cluster_name>  |
|--|--|
| 覆盖已配置的集群名称。 |  |

*Arguments*

|CLUSTER_CONFIG_FILE|  |
|--|--|
|需要的参数  |  |

**ray exec**
进入ray集群中

    ray exec [OPTIONS] CLUSTER_CONFIG_FILE CMD

*Options*

|--docker |
|--|--|
|  在cluster_config中指定的docker容器中运行命令。|  |

|--stop| 
|--|--|
|命令运行完后停止集群。  |  |

| --start  |
|--|--|
|如果需要，启动集群。  |  |

|--screen  |
|--|--|
|在屏幕上运行该命令。  |  |

|--tmux  |
|--|--|
| 在tmux中运行该命令。 |  |

|-n, --cluster-name <cluster_name>  |
|--|--|
|  覆盖已配置的集群名称。|  |

| --port-forward <port_forward>  |
|--|--|
| 端口转发。 |  |

*Arguments*

|CLUSTER_CONFIG_FILE  |
|--|--|
|需要的参数  |  |

|CMD  |
|--|--|
| 需要的参数 |  |

**ray attach**
进入ray集群中

```
ray attach [OPTIONS] CLUSTER_CONFIG_FILE
```
*Options*

|--start|  
|-------|--|
|     如果需要，启动集群。  |  |

|--tmux  |
|--|--|
| 在tmux中运行该命令。 |  |

|-n, --cluster-name <cluster_name>  |
|--|--|
|覆盖已配置的集群名称。  |  |

|-N, --new  |
|--|--|
| 强制创建一个新屏幕 |  |

*Arguments*
|CLUSTER_CONFIG_FILE  |
|--|--|
|需要的参数  |  |

**ray get_head_ip**
获取头节点ip地址

```
ray get_head_ip [OPTIONS] CLUSTER_CONFIG_FILE
```
*Options*

|-n, --cluster-name <cluster_name>  |
|--|--|
|覆盖已配置的集群名称。  |  |

*Arguments*

|CLUSTER_CONFIG_FILE  |
|--|--|
| 需要的参数 |  |

**ray stack**

```
ray stack [OPTIONS]
```

**ray timeline**

```
ray timeline [OPTIONS]
```
*Options*

| --redis-address <redis_address>  |
|--|--|
| 覆盖要连接的redis地址 |  |


此篇主要参考Ray官网，如有错误，请阅读者提出指正，谢谢！
原英文链接：https://ray.readthedocs.io/en/latest/api.html     
ray综述介绍：https://blog.csdn.net/weixin_43255962/article/details/88689665
