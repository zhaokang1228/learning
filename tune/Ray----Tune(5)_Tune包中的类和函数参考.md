本片主要介绍一下tune中常用的一些函数用处，可以作为一个简单的API使用。
## ray.tune
| ray.tune.grid_search(values) |   
|--|
用于指定值上的网格搜索的快捷方法。

    参数：
        	values –参数将被网格化的迭代器。

|ray.tune.register_env(name, env_creator)  | 
|:--|
注册一个自定义环境，以便与RLlib一起使用。

    参数：
    		name (str) –注册的名称。
    		env_creator (obj) –创建环境的函数。

| ray.tune.register_trainable(name, trainable) |  
|:--|
注册一个可训练的（trainable）函数或类。

    参数：
    	name (str) –注册的名称。
    	trainable (obj) – 函数或tune的可训练类。函数必须以(config, status_reporter)作为参数，
    		并在注册期间自动转换为类。

### ray.tune.run
|[ray.tune.run(run_or_experiment, name=None, stop=None, config=None, resources_per_trial=None, num_samples=1, local_dir=None, upload_dir=None, trial_name_creator=None, loggers=None, sync_function=None, checkpoint_freq=0, checkpoint_at_end=False, export_formats=None, max_failures=3, restore=None, search_alg=None, scheduler=None, with_server=False, server_port=4321, verbose=2, resume=False, queue_trials=False, reuse_actors=False, trial_executor=None, raise_on_failed_trial=True)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/tune.html#run)  |  
|:--|
运行训练。

```
参数：
		run_or_experiment (function|class|str|Experiment) －如果是function | class | str，
			那么这就是要训练的算法或模型。 这可以指内置算法的名称（例如RLLib的DQN或PPO），
			用户定义的Trainable函数或类，或者在tune注册表中注册的Trainable函数或类的字符串标
			识符。 如果是Experiment，那么Tune将根据Experiment.spec执行训练。
		name (str) – 实验名称。
		stop (dict) – 停止条件。键可以是' train() '返回结果中的任何字段，以先到达的字段为准。
			默认为空dict。
		config (dict) – 特定算法的Tune variant generation （变量生成器）的配置信息。
			默认为空字典。自定义搜索算法可能会忽略这一点。
		resources_per_trial (dict) -每次试验分配的机器资源，如{“cpu”:64，“gpu”:8}。注意，
			除非在这里指定gpu，否则不会分配GPU资源。Trainable.default_resource_request()中默认
			为1个CPU和0个gpu。
		num_samples (int) –从超参数空间采样的次数。默认为1。如果grid_search作为参数提供，
			则网格将多次重复使用num_samples。
		local_dir (str) –保存训练结果到本地路径。默认是~/ray_results。
		upload_dir (str) –可选URI将训练结果同步到(例如s3://bucket)。
		trial_name_creator (func) –用于生成训练字符串表示形式的可选函数。
		loggers (list) – 每次试验都要使用的日志记录器创建者列表。如果没有，
			默认为ray.tune.logger.DEFAULT_LOGGERS. 见ray/tune/logger.py.
		sync_function (func|str) –函数的作用是:将local_dir同步到upload_dir。如果是字符串，那么它
			必须是一个字符串模板，以便syncer运行。如果没有提供，
			sync命令默认为标准S3或gsutil sync comamnds。
		checkpoint_freq (int) – 检查点之间的训练迭代次数。值0(默认值)禁用检查点。
		checkpoint_at_end (bool) – 是否在实验结束时进行检查点检查，而不考虑checkpoint t_freq。
			默认是假的。
		export_formats (list) – 实验结束时导出的格式列表。默认是没有的。
		max_failures (int) –至少多次尝试从最后一个检查点恢复试验。仅当启用检查点时才适用。
			设置为-1将导致无限的恢复重试。默认为3。
		restore (str) – 路径监测点。只有在运行1次试验时设置才有意义。默认为没有。
		search_alg (SearchAlgorithm) – 搜索算法。默认为BasicVariantGenerator。
		scheduler (TrialScheduler) –用于执行实验的调度程序。选择FIFO(默认值)、medianstop、
			AsyncHyperBand和HyperBand。
		with_server (bool) –启动后台Tune服务器。需要使用客户端API。
		server_port (int) –启动TuneServer的端口号。
		verbose (int) –0，1或 2冗长的模式。0 =静音，1 =只更新状态，2 =状态和试验结果。
		resume (bool|"prompt") –如果检查点存在，实验将从那里继续进行。
			如果resume是“prompt”，那么如果检测到检查点，Tune将会提示。
		queue_trials (bool) – 当集群当前没有足够的资源来启动一个测试时，是否对测试进行排队。
			在自动缩放集群上运行时，应将此设置为True，以启用自动缩放。
		reuse_actors (bool) –如果可能，是否在不同的试验之间重用actor。这可以极大地加快经常启动
			和停止参与者的实验(例如，在时间复用模式下的PBT)。这要求试验具有相同的资源需求。
		trial_executor (TrialExecutor) –管理训练的执行。
		raise_on_failed_trial (bool) –当实验完成时，如果存在失败的试验(错误状态)，
			则提出TuneError。
		
返回：
		训练对象列表。
		
```

例子：

```
>>> tune.run(mytrainable, scheduler=PopulationBasedTraining())
```

```
>>> tune.run(mytrainable, num_samples=5, reuse_actors=True)
```

```
>>> tune.run(
        "PG",
        num_samples=5,
        config={
            "env": "CartPole-v0",
            "lr": tune.sample_from(lambda _: np.random.rand())
        }
    )
```

### ray.tune.run_experiments
|[ray.tune.run_experiments(experiments, search_alg=None, scheduler=None, with_server=False, server_port=4321, verbose=2, resume=False, queue_trials=False, reuse_actors=False, trial_executor=None, raise_on_failed_trial=True)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/tune.html#run_experiments)|  
|:--|

运行和阻塞，直到所有的试验结束

例子：

```
>>> experiment_spec = Experiment("experiment", my_func)
>>> run_experiments(experiments=experiment_spec)
```

```
>>> experiment_spec = {"experiment": {"run": my_func}}
>>> run_experiments(experiments=experiment_spec)
```

```
>>> run_experiments(
>>>     experiments=experiment_spec,
>>>     scheduler=MedianStoppingRule(...))
```

```
>>> run_experiments(
>>>     experiments=experiment_spec,
>>>     search_alg=SearchAlgorithm(),
>>>     scheduler=MedianStoppingRule(...))
```

    返回：
    	试验对象列表，包含每个已执行试验的数据。
	
### class ray.tune.Experiment
|  [class ray.tune.Experiment(name, run, stop=None, config=None, resources_per_trial=None, num_samples=1, local_dir=None, upload_dir=None, trial_name_creator=None, loggers=None, sync_function=None, checkpoint_freq=0, checkpoint_at_end=False, keep_checkpoints_num=None, checkpoint_score_attr=None, export_formats=None, max_failures=3, restore=None, repeat=None, trial_resources=None, custom_loggers=None)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/experiment.html#Experiment)|  
|:--|
跟踪试验规范。如果需要，隐式地注册可训练对象。
例子：

```
>>> experiment_spec = Experiment(
>>>     "my_experiment_name",
>>>     my_func,
>>>     stop={"mean_accuracy": 100},
>>>     config={
>>>         "alpha": tune.grid_search([0.2, 0.4, 0.6]),
>>>         "beta": tune.grid_search([1, 2]),
>>>     },
>>>     resources_per_trial={
>>>         "cpu": 1,
>>>         "gpu": 0
>>>     },
>>>     num_samples=10,
>>>     local_dir="~/ray_results",
>>>     upload_dir="s3://your_bucket/path",
>>>     checkpoint_freq=10,
>>>     max_failures=2)
```

**classmethod from_json(name, spec)** 


从JSON生成一个实验对象。

    参数：
    		name (str) – 实验的名称。
    		spec (dict) –JSON配置的实验。
### class ray.tune.function
|  [class ray.tune.function(func)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/variant_generator.html#function)  |  
|:--|
封装func，以确保在解析期间不展开它。
必须通过在`tune.sample_from()`或`tune.function()`中包装函数来消除tune配置中函数参数的使用歧义。

    参数：
    		func – 要封装的函数名称。

### class ray.tune.sample_from
| [class ray.tune.sample_from(func)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/variant_generator.html#sample_from) |  
|:--|
指定该tune应该从该函数取样配置值。
必须通过在`tune.sample_from()`或`tune.function()`中包装函数来消除tune配置中函数参数的使用歧义。

    参数：
    		func – 一个可调用的函数，用来绘制样本。
### class ray.tune.Trainable
| [class ray.tune.Trainable(config=None, logger_creator=None)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/trainable.html#Trainable) |  
|:--|
用于可训练模型、函数等的抽象类。

对一个可训练对象调用`_train()`将执行一个训练的逻辑迭代。根据经验，一次训练（trial）调用执行时间应该足够长，以避免开销(即超过几秒)，但要足够短，以便定期报告进度(即最多几分钟)。

调用`_save()`应该将可训练对象的训练状态保存到磁盘，而`_restore(path)`应该将可训练对象恢复到给定的状态。

通常，当子类化Trainable时，只需要在这里实现`_train`、`_save`和`_restore`。

**注意** ，如果您不需要检查点/恢复功能，那么您也可以只向配置提供一个my_train(config, reporter)函数，而不是实现这个类。该函数将自动转换为该接口(没有检查点功能)。

**classmethod default_resource_request(config)**
返回给定配置的资源需求。
这可以通过子类覆盖来设置正确的试验资源分配，因此用户不需要这样做。

**classmethod resource_help(config)**
返回配置此可训练资源的帮助字符串。

**train()**
运行一个训练的逻辑迭代。
子类应该覆盖_train()来返回结果。这个类自动填充结果中的以下字段:

 - done (bool): 训练是终止。只有在没有提供时才填写。
 - time_this_iter_s (float):此迭代运行所需的时间(以秒为单位)。可能会覆盖系统计算的时间差。
 - time_total_s (float):整个实验的累积时间以秒为单位。
 - experiment_id (str):此实验的唯一字符串标识符。此id在检查点/恢复调用之间保存。
 - training_iteration (int): 此训练迭代的索引，例如调用train()。
 - pid (str): 训练过程的id。
 - date (str): 处理结果的格式化日期。
 - timestamp (str): 处理结果时的UNIX时间戳
 - hostname (str):承载培训练过程的机器的主机名。
 - node_ip (str): 培训过程的主机节点ip。

返回：描述培训进展的词典。

**delete_checkpoint(checkpoint_dir)**
删除checkpoint_folder:param checkpoint_dir::type checkpoint_dir: path to checkpoint中的子目录

**save(checkpoint_dir=None)**

将当前模型状态保存到检查点。
子类应该覆盖`_save()`来保存状态。此方法在保存的路径旁边转储其他元数据。

    参数：
    		checkpoint_dir (str) –可选dir来放置检查点。
    返回：
    		可以传递给 restore() 的检查点路径。

**save_to_object()**
将当前模型状态保存到Python对象中。它也保存到磁盘，但不返回检查点路径。

    返回：
    		持有检查点数据的对象。

**restore(checkpoint_path)**
从给定的模型检查点恢复训练状态。
这些检查点从save()调用返回。
子类应该覆盖`_restore()`来恢复状态。此方法恢复检查点保存的其他元数据。

**restore_from_object(obj)**
从检查点对象恢复训练状态。
这些检查点从调用`save_to_object()`返回。

**export_model(export_formats, export_dir=None)**
基于`export_formats`的导出模型。
子类应该覆盖`_export_model()`来将模型实际导出到本地目录。

```
参数：
		export_formats (list) –应该导出的格式列表。
		export_dir (str) –可选dir来放置导出的模型。默认为self.logdir。
返回：
		将导出格式映射到成功导出的模型的dict。
```

**reset_config(new_config)**
在不重新启动试验的情况下重置配置。
此方法是可选的，但是可以实现来加速算法(如PBT)，并允许性能优化(如使用reuse_actors=True运行实验)。

```
参数：
		new_config (dir) –更新了可训练的超参数配置。
返回：
		如果重置成功，则为真，否则为假
```

**stop()**
释放此可训练程序使用的所有资源。

**_train()**
子类应该覆盖它来实现train()。

```
返回：
		描述培训进展的词典。
```

**_save(checkpoint_dir)**
子类应该覆盖它来实现save()。

```
参数：
		checkpoint_dir (str) –检查点文件必须存储的目录。
返回：
		如果是字符串，返回值应该是传递给_restore()的检查点路径。如果是dict，返回值将通过调优自动序列化并传递给_restore()。
返回类型：
		检查点(str | dict)
```

例子:

```
>>> print(trainable1._save("/tmp/checkpoint_1"))
"/tmp/checkpoint_1/my_checkpoint_file"
>>> print(trainable2._save("/tmp/checkpoint_2"))
{"some": "data"}
```

**_restore(checkpoint)**
子类应该覆盖它来实现restore()。

```
参数：
		checkpoint (str | dict) –该值由_save返回。如果是字符串，那么它就是检查点路径。
```

**_setup(config)**
子类应该覆盖它以进行自定义初始化。

```
参数：
		config (dict) –给出超参数和其他配置。self.config副本。
```

**_log_result(result)**
子类可以选择性地覆盖此选项以自定义日志记录。

```
参数：
		result (dict) –由_train()返回的训练结果。
```

**_stop()**
子类应该覆盖此选项在终止时清理所有。

**_export_model(export_formats, export_dir)**
子类应该覆盖这个来导出模型。

```
参数：
		export_formats (list) – 应该导出的格式列表。
		export_dir (str) –目录来放置导出的模型。
返回：
		将导出格式映射到成功导出的模型的dict。
```
### class ray.tune.function_runner

|  [class ray.tune.function_runner.StatusReporter(result_queue, continue_semaphore)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/function_runner.html#StatusReporter)   |  
|:--|
对象传递到您的函数中，您可以通过该函数报告状态。

例子：

```
>>> def trainable_function(config, reporter):
>>>     assert isinstance(reporter, StatusReporter)
>>>     reporter(timesteps_this_iter=1)
```

__ call __ (**kwargs) 
报告最新的培训状况。
当培训工作完成时，请提交`done=True` 。

    参数：
    		kwargs –最新培训结果状态。

例子：

```
>>> reporter(mean_accuracy=1, training_iteration=4)
>>> reporter(mean_accuracy=1, training_iteration=4, done=True)
```

    可能报错：
    		StopIteration – 如果已发出停止试验的信号，则会引发StopIteration异常。

## ray.tune.schedulers
### TrialScheduler
|   [class ray.tune.schedulers.TrialScheduler](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/trial_scheduler.html#TrialScheduler)    |  
|:--|

用于实现试用调度程序类的接口。

**CONTINUE = 'CONTINUE'**
继续试验执行的状态

**PAUSE = 'PAUSE'**
暂停试验执行的状态

**STOP = 'STOP'**
终止试验执行的状态。

**on_trial_add(trial_runner, trial)**
将新试验添加到试验运行程序时调用。

**on_trial_error(trial_runner, trial)**
试验错误通知。
只有当试验处于运行（CONTINUE）状态时才会调用这个函数。

**on_trial_result(trial_runner, trial, result)**
返回试验的每个中间结果（状态）。
此时，试用调度程序可以通过返回CONTINUE、PAUSE和STOP中的一个来做出决策。只有当试验处于运行状态时才会调用这个函数。

**on_trial_complete(trial_runner, trial, result)**
完成试验的通知。
只有当试验处于运行状态并自然完成或手动终止时，才会调用此函数。

**on_trial_remove(trial_runner, trial)**
移除试验。
当试验处于暂停或挂起状态时调用。否则,调用`on_trial_complete`。

**choose_trial_to_run(trial_runner)**
选择一个新的试验去运行。
这应该返回处于挂起或暂停状态的trial_runner中的一个试验。这个函数必须是幂等的。
如果试验没有就绪，返回空。

**debug_string()**
返回用于打印到控制台的人类可读消息。

### HyperBandScheduler
|  [class ray.tune.schedulers.HyperBandScheduler(time_attr='training_iteration', reward_attr='episode_reward_mean', max_t=81)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/hyperband.html#HyperBandScheduler)   |  
|:--|

基础类：`ray.tune.schedulers.trial_scheduler.FIFOScheduler`
实现 HyperBand 提前停止算法。
HyperBandScheduler 早期停止使用 HyperBand 优化算法的试验。它将试验分为不同大小的括号，并定期在每个括号内提前停止性能较差的试验。

要使用这个带tune的HyperBand 实现，您所需要做的就是指定一个试验可以运行`max_t`的最大时间长度、时间单位`time_attr`和报告的目标值`reward_attr`的名称。我们根据给定的值自动确定其他超带参数的合理值。

例如，根据`episode_mean_reward` attr将试验限制在10分钟内并提前停止，构造如下:

```
HyperBand('time_total_s', 'episode_reward_mean', max_t=600)
```
注意，Tune的停止标准将与HyperBand的早期停止机制一起应用。

更多详情请参考： https://people.eecs.berkeley.edu/~kjamieson/hyperband.html


```
参数：
		time_attr (str) – 训练结果用于比较时间。注意，您可以传递一些非时间的东西，比如
			training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
		reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使
			用此属性。
		max_t (int) –每次试验的最大时间单位。max_t时间单位(由time_attr决定)通过后，将停止试验。调度
			程序将在此时间通过后终止试验。注意，这与原始HyperBand论文中提到的max_t的语义不同。

```

**on_trial_add(trial_runner, trial)**
增加了新的试验。
在一个新的试用程序中添加,如果当前括号没有填充,则添加到当前括号。否则,如果没有填充流,创建新的括号,再添加到当前括号。另外,创建新的迭代,创建新的括号,添加到括号。

**on_trial_result(trial_runner, trial, result)**
如果括号完成，所有的试验将停止

如果给定的试验结束，并且没有完成括号迭代，则将暂停试验并放弃资源。

此调度程序将不启动试验，而是停止试验。当前正在运行的试验将不被处理，因为试验运行器将被赋予处理它的控制权。

**on_trial_remove(trial_runner, trial)**
试验终止时的通知。

从括号中删除试验信息。如果括号没有完成，则触发对半。

**on_trial_complete(trial_runner, trial, result)**
如果试验提前完成，清除括号里的试验信息。

**on_trial_error(trial_runner, trial)**
如果试验早期出错，从括号中清除试验信息。

**choose_trial_to_run(trial_runner)**
按完成百分比在迭代内公平调度。
未使用的试验列表，因为所有试验都作为调度程序状态跟踪。如果迭代被占用(即没有要运行的试验)，那么查看下一个迭代。

**debug_string()**
这为算法提供了一个进度通知。
对于每个括号，算法将输出如下字符串:

```
Bracket(Max Size (n)=5, Milestone (r)=33, completed=14.6%): {PENDING: 2, RUNNING: 3, TERMINATED: 2}

```
“Max Size”表示根据 Hyperband 算法设置的待定/运行实验的最大数量。
“Milestone” 表示在下一次减半之前，试验将运行的迭代。
“Completed”表示一个近似的进度度量。有些括号，比如未填充的，不会达到100。

### AsyncHyperBandScheduler
|  [class ray.tune.schedulers.AsyncHyperBandScheduler(time_attr='training_iteration', reward_attr='episode_reward_mean', max_t=100, grace_period=10, reduction_factor=3, brackets=3)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/async_hyperband.html#AsyncHyperBandScheduler)  |  
|:--|

基础类：`ray.tune.schedulers.trial_scheduler.FIFOScheduler`

实现Async Successive减半。
这应该提供与超带类似的理论性能，但是避免了HyperBand 所面临的掉队问题。一个实现细节是，当使用多个括号时，试验分配到括号是随机进行的，具有超过软最大值的概率。

更多详情参考：https://openreview.net/forum?id=S1Y7OOlRZ

```
参数：
		time_attr (str) – 用于比较时间的训练结果。注意，您可以传递一些非时间的东西，比如
				training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
		reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使用
			此属性。
		max_t (float) – 每次训练的最大时间单位。max_t时间单位(由time_attr决定)通过后，将停止试验。
		grace_period (float) – 至少在这个时候停止试验。这些单元与time_attr所命名的属性相同。
		reduction_factor (float) –用于设定减半率和用量。这只是一个无单位标量。
		brackets (int) –括号数量。 每个括号具有不同的减半率，由减少系数指定。

```

**on_trial_add(trial_runner, trial)**
将新试验添加到试验运行程序时调用。

**on_trial_result(trial_runner, trial, result)**
调用由试验返回的每个中间结果。
此时，试用调度程序可以通过返回CONTINUE、PAUSE和STOP中的一个来做出决策。只有当试验处于运行状态时才会调用这个函数。

**on_trial_complete(trial_runner, trial, result)**
完成试验的通知。
只有当试验处于运行状态并自然完成或手动终止时，才会调用此函数。

**on_trial_remove(trial_runner, trial)**
移除试验。
当试验处于暂停或挂起状态时调用。否则,调用on_trial_complete。

**debug_string()**
返回用于打印到控制台的人类可读消息。

### MedianStoppingRule
|     [class ray.tune.schedulers.MedianStoppingRule(time_attr='time_total_s', reward_attr='episode_reward_mean', grace_period=60.0, min_samples_required=3, hard_stop=True, verbose=True)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/median_stopping_rule.html#MedianStoppingRule) |  
|:--|

基础类：`ray.tune.schedulers.trial_scheduler.FIFOScheduler`
实现[Vizier论文](https://research.google.com/pubs/pub46180.html)中描述的中值停止规则。

```
参数：
		time_attr (str) –训练结果用于比较时间。注意，您可以传递一些非时间的东西，比如
			training_iteration作为进度的度量，惟一的要求是属性应该单调地增加。
		reward_attr (str) –训练结果目标值属性。与time_attr一样，这可以引用任何应该随时间增长的客
	观值。
		grace_period (float) – 至少在这个时候停止试验。这些单元与time_attr所命名的属性
	相同。
		min_samples_required (int) –计算中值的最小样本。
		hard_stop (bool) – 如果错误，暂停试验而不是停止试验。当所有其他试验完成时，暂停的试验将重
				新开始并允许运行FIFO。
		verbose (bool) – 如果为真，将在每次试验报告时输出中值和最佳结果。默认值为True。

```

**on_trial_result(trial_runner, trial, result)**
用于早期停止的回调。
如果t阶段试验的最佳目标值严格小于t阶段之前报告的所有已完成试验目标的运行平均值的中值，则此停止规则将停止运行试验。

**on_trial_complete(trial_runner, trial, result)**
完成试验的通知。
只有当试验处于运行状态并自然完成或手动终止时，才会调用此函数。

**on_trial_remove(trial_runner, trial)**
如果已暂停并已运行过，则将试用标记为已完成。

**debug_string()**
返回用于打印到控制台的人类可读消息。

### FIFOScheduler
|   [class ray.tune.schedulers.FIFOScheduler](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/trial_scheduler.html#FIFOScheduler)     |  
|:--|

基础类：`ray.tune.schedulers.trial_scheduler.TrialScheduler`
简单的调度程序，只是运行在提交顺序试验。

**on_trial_add(trial_runner, trial)**
将新试验添加到试验运行程序时调用。

**on_trial_error(trial_runner, trial)**
试验错误通知。
只有当试验处于运行状态时才会调用这个函数。

**on_trial_result(trial_runner, trial, result)**
调用由试验返回的每个中间结果。
此时，试用调度程序可以通过返回CONTINUE、PAUSE和STOP中的一个来做出决策。只有当试验处于运行状态时才会调用这个函数。

**on_trial_complete(trial_runner, trial, result)**
完成试验的通知。
只有当试验处于运行状态并自然完成或手动终止时，才会调用此函数。

**on_trial_remove(trial_runner, trial)**
移除试验。
当试验处于暂停或挂起状态时调用。否则,调用`on_trial_complete`。

**choose_trial_to_run(trial_runner)**
选择一个新的试验去运行。
这应该返回处于挂起或暂停状态的trial_runner中的一个试验。这个函数必须是幂等的。
如果没有准备好试验，就返回none。

**debug_string()**
返回用于打印到控制台的人类可读消息。

### PopulationBasedTraining
|     [class ray.tune.schedulers.PopulationBasedTraining(time_attr='time_total_s', reward_attr='episode_reward_mean', perturbation_interval=60.0, hyperparam_mutations={}, resample_probability=0.25, custom_explore_fn=None)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/schedulers/pbt.html#PopulationBasedTraining)   |  
|:--|

基础类：`ray.tune.schedulers.trial_scheduler.FIFOScheduler`
实现了[基于种群的训练(PBT)算法](https://deepmind.com/blog/population-based-training-neural-networks)。
PBT并行地训练一组模型(或代理)。性能较差的模型会周期性地克隆性能最好的模型的状态，并对它们的超参数进行随机突变，希望能够超过当前性能最好的模型。

与其他超参数搜索算法不同，PBT在训练过程中会改变超参数。这支持非常快的超参数发现，并自动发现良好的退火调度。
此调优PBT实现将所有添加的试验视为PBT总体的一部分。如果试验的数量超过了集群的能力，将对试验进行时间复用，以平衡整个人群的培训进展。

```
参数:
		time_attr (str) –训练结果用的比较时间。注意，您可以传递一些非时间的东西，比如training_iteration
				作为进度的度量，惟一的要求是属性应该单调地增加。
		reward_attr (str) – 训练结果目标值属性。与time_attr一样，这可以引用任何客观值。停止过程将使用	
				此属性。
		perturbation_interval (float) – 模型将考虑在这个 time_attr时间间隔内的扰动。注意，扰动会导致检查
				点开销，因此不应该将此设置为太频繁。
		hyperparam_mutations (dict) –Hyperparams变异。格式如下:对于每个键，可以提供列表或函数。列
				表	指定一组允许的分类值。函数指定连续参数的分布。您必须指定至少一个
				hyperparam_variables或	custom_explore_fn。
		resample_probability (float) – 当应用hyperparam_mutation时，从原始分布重新采样的概率。如果不
				进行重采样，则连续的值会受到1.2或0.8倍的扰动，离散的值则会变为相邻值。
		custom_explore_fn (func) –您还可以指定一个定制的探测函数。在应用了hyperparam_variables的内
				置扰动之后，这个函数被调用为f(config)，并且应该返回根据需要更新的配置。您必须指定至少
				一个hyperparam_variables或custom_explore_fn。

```

例子：

```
>>> pbt = PopulationBasedTraining(
>>>     time_attr="training_iteration",
>>>     reward_attr="episode_reward_mean",
>>>     perturbation_interval=10,  # every 10 `time_attr` units
>>>                                # (training_iterations in this case)
>>>     hyperparam_mutations={
>>>         # Perturb factor1 by scaling it by 0.8 or 1.2. Resampling
>>>         # resets it to a value sampled from the lambda function.
>>>         "factor_1": lambda: random.uniform(0.0, 20.0),
>>>         # Perturb factor2 by changing it to an adjacent value, e.g.
>>>         # 10 -> 1 or 10 -> 100. Resampling will choose at random.
>>>         "factor_2": [1, 10, 100, 1000, 10000],
>>>     })
>>> run_experiments({...}, scheduler=pbt)
```
**on_trial_add(trial_runner, trial)**
将新试验添加到试验运行程序时调用。

**on_trial_result(trial_runner, trial, result)**
调用由试验返回的每个中间结果。
此时，试用调度程序可以通过返回CONTINUE、PAUSE和STOP中的一个来做出决策。只有当试验处于运行状态时才会调用这个函数。

**choose_trial_to_run(trial_runner)**
确保所有试验获得公平的时间分配(由time_attr定义)。
这使得PBT调度程序能够支持比任何给定时间集群中所能容纳的更多的并发测试。

**debug_string()**
返回用于打印到控制台的人类可读消息。

## ray.tune.suggest
### SearchAlgorithm
|   [class ray.tune.suggest.SearchAlgorithm](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/search.html#SearchAlgorithm)    |  
|:--|

用于超参数搜索的事件处理程序API的接口。
与试用调度器不同，搜索算法将没有修改执行的能力（例如停止和暂停试验)。
手动添加的试验(即，通过客户端API)也会在新的事件发生时通知这个类，因此自定义搜索算法应该维护一个由这个类生成的trials ID列表。
更多请参考： `ray.tune.suggest.BasicVariantGenerator.`

**add_configurations(experiments)**
跟踪给定的实验说明。

```
参数：
		experiments (Experiment | list | dict) –运行的实验。
```
**next_trials()**

提供要排队进入试用运行器的试验对象。

```
返回：试验队列。
返回类型：试验（list）。
```

**on_trial_result(trial_id, result)**
调用由试验返回的每个中间结果。
只有当试验处于运行状态时才会调用这个函数。

```
参数：
		trial_id –试验的id。
```

**on_trial_complete(trial_id, result=None, error=False, early_terminated=False)**
完成试验的通知。

```
参数：
		trial_id – 试验id。
		result (dict) –默认为没有。当试验处于运行状态并自然完成或手动终止时，将向dict提供此通知。
		error (bool) – 默认值为False。如果试验处于运行状态且出现错误，则为真。
		early_terminated (bool) –默认值为False。如果试验在暂停或挂起状态下停止，则为真。
```

**is_finished()**
如果没有剩余的试验排队进入TrialRunner，则返回True。
可以在所有试验完成之前返回True。

### BasicVariantGenerator
|     [class ray.tune.suggest.BasicVariantGenerator(shuffle=False)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/basic_variant.html#BasicVariantGenerator)      |  
|:--|
基础类：`ray.tune.suggest.search.SearchAlgorithm`
使用Tune的变体生成来解析变量。
更多请看：`ray.tune.suggest.variant_generator`.
例子:

```
>>> searcher = BasicVariantGenerator()
>>> searcher.add_configurations({"experiment": { ... }})
>>> list_of_trials = searcher.next_trials()
>>> searcher.is_finished == True
```

**add_configurations(experiments)**
给出了实验规格说明。

```
参数：
		experiments (Experiment | list | dict) – 运行的实验。
```

**next_trials()**
提供要排队进入试验运行器的试用对象。

```
返回：
		试验列表。
返回类型：
		试验（list）。
```

**is_finished()**
如果没有剩余的试验排队进入TrialRunner，则返回True。
可以在所有试验完成之前返回True。

### SuggestionAlgorithm
|   [class ray.tune.suggest.SuggestionAlgorithm](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/suggestion.html#SuggestionAlgorithm)  |  
|:--|
基础类：ray.tune.suggest.search.SearchAlgorithm
用于基于建议（ suggestion-based ）的算法的抽象类。
自定义搜索算法可以通过覆盖为试验提供生成参数的`_suggest`方法轻松扩展该类
为了跟踪建议及其相应的评估，方法`_suggest`将被传递一个trial_id，该id将在后续通知中使用。

例子：

```
>>> suggester = SuggestionAlgorithm()
>>> suggester.add_configurations({ ... })
>>> new_parameters = suggester._suggest()
>>> suggester.on_trial_complete(trial_id, result)
>>> better_parameters = suggester._suggest()
```

**add_configurations(experiments)**
给出了实验规格说明。

```
参数：
		experiments (Experiment | list | dict) –运行的试验。
```

**next_trials()**
提供要排队进入试用运行器的一批试验对象。
当self_trial_generator返回None时，批处理结束。

```
返回：
		试验列表。
返回类型：
		试验（list）。
```
** _generate_trials(experiment_spec, output_path='')**
从_suggest生成配置试验。
创建一个传递给_suggest的trial_id。

```
Yields：
		创建一个传递给_suggest的trial_id。
```

**is_finished()**
如果没有剩余的试验排队进入TrialRunner，则返回True。
可以在所有试验完成之前返回True。

** _suggest(trial_id)**
查询算法以检索下一组参数。

```
参数：
		trial_id –用于后续通知的试验ID。
返回：
		如果可能，为试验配置。否则，返回None，这将暂时停止TrialRunner的查询。
返回类型：
		dict|None
```

例子：

```
>>> suggester = SuggestionAlgorithm(max_concurrent=1)
>>> suggester.add_configurations({ ... })
>>> parameters_1 = suggester._suggest()
>>> parameters_2 = suggester._suggest()
>>> parameters_2 is None
>>> suggester.on_trial_complete(trial_id, result)
>>> parameters_2 = suggester._suggest()
>>> parameters_2 is not None
```


## ray.tune.logger

|   [class ray.tune.logger.Logger(config, logdir, upload_uri=None)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/logger.html#Logger)     |  
|:--|

ray.tune的日志记录接口。
默认情况下，使用UnifiedLogger实现，它可以同时记录多种格式的结果(TensorBoard、rllab/viskit、纯json、定制的日志记录器)。

```
参数：
		config – 配置传递给所有日志程序创建者。
		logdir –所有日志程序创建者要登录的目录。
		upload_uri (str) - 可选URI，其中logdir被同步到。
```

**on_result(result)**
给定一个结果，将其追加到现有日志中。


**close()**
释放此日志程序使用的所有资源。

**flush()**
刷新所有磁盘写入到存储器。


































