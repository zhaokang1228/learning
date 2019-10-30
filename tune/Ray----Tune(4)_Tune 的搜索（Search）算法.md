Tune提供了很多有效的优化模型的超参搜索算法。Tune允许您结合使用不同的搜索算法和不同的试验调度程序。在默认情况下，Tune将隐式地使用Variant Generation算法来创建试验。
你可以利用这些搜索算法如下:

```
tune.run(my_function, search_alg=SearchAlgorithm(...))
```
目前，Tune提供了以下搜索算法(和库集成):

 - 1.Grid Search and Random Search（网格搜索和随机搜索）
 - 2.BayesOpt（贝叶斯优化）
 - 3,HyperOpt（超参优化）
 - 4.SigOpt
 - 5.Nevergrad
 - 6.Scikit-Optimize
 - 7.GeneticSearch

注：目前，所有调度程序都采用一个度量标准metric，该度量标准是您的Trainable的结果字典中返回的值，并且根据模式mode而最大化或最小化。(老版本中是所有搜索算法都接受一个reward_attr，即目标参数标准，该值假定为最大化。)。由于此篇写的比较早，故后边的例子主要是按照老版本进行说明的。

#### １．Grid Search and Random Search（网格搜索和随机搜索）
默认情况下，Tune使用默认的搜索空间和变量生成（Variant Generation）过程来创建和排队trail(试验队列)。这支持随机搜索和网格搜索作为`tune.run`的指定的`config`参数。

| [class ray.tune.suggest.BasicVariantGenerator(shuffle=False)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/basic_variant.html#BasicVariantGenerator) |  
|--|

基础：`ray.tune.suggest.search.SearchAlgorithm`
使用Tune的变量生成（variant generation）去分析变量。
其他详情请看：`ray.tune.suggest.variant_generator.`
例子：

```
>>> searcher = BasicVariantGenerator()
>>> searcher.add_configurations({"experiment": { ... }})
>>> list_of_trials = searcher.next_trials()
>>> searcher.is_finished == True
```
注意，其他搜索算法不一定要扩展该类，可能需要与默认tune格式不同的搜索空间声明。

#### ２．BayesOpt Search（贝叶斯优化搜索）
`BayesOptSearch`是一种由 [bayesian-optimization](https://ray.readthedocs.io/en/latest/tune-searchalg.html)包支持的搜索算法，用于执行 **基于顺序模型的超参数优化** 。注意，这个类没有扩展`ray.tune.suggest`。因此，在使用BayesOptSearch时，您将无法使用Tune的默认变量生成/搜索空间声明。
要用这个BayesOpt  搜索算法，首先需要通过命令安装 bayesian-optimization包：

```
pip install bayesian-optimization
```
该算法需要[设置搜索空间和定义实用函数](https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb)。你可以使用BayesOptSearch如下:

```
tune.run(... , search_alg=BayesOptSearch(bayesopt_space, utility_kwargs=utility_params, ... ))
```
一个例子可以在[bayesopt_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/bayesopt_example.py)中找到。

| [class ray.tune.suggest.bayesopt.BayesOptSearch(space, max_concurrent=10, reward_attr='episode_reward_mean', utility_kwargs=None, random_state=1, verbose=0, **kwargs)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/bayesopt.html#BayesOptSearch) |  
|--|

基础：`ray.tune.suggest.suggestion.SuggestionAlgorithm`
BayesOpt的一个包装器提供试验建议。

```
参数：
	space (dict) –连续的搜索空间。参数将从该空间取样，用于运行试验。
	max_concurrent (int) –最大同时试验次数。默认为10。
	reward_attr (str) –训练结果目标值属性。这指的是一个递增的值。
	utility_kwargs (dict) – 参数来定义实用函数。必须为键类型、kappa和xi提供值。
	random_state (int) –用于初始化BayesOpt。
	verbose (int) – 设置BayesOpt包的复杂级别。
```
例子：

```
>>> space = {
>>>     'width': (0, 20),
>>>     'height': (-100, 100),
>>> }
>>> algo = BayesOptSearch(
>>>     space, max_concurrent=4, reward_attr="neg_mean_loss")
```
#### ３．HyperOpt Search (Tree-structured Parzen Estimators)
`HyperOptSearch`是一种由[HyperOpt](http://hyperopt.github.io/hyperopt)支持的搜索算法，用于执行 **基于顺序模型的超参数优化** 。注意，这个类没有扩展`ray.tune.suggest.BasicVariantGenerator`。因此，在使用HyperOptSearch时，将无法使用Tune的默认变量生成/搜索空间声明。
为了使用这个搜索算法，你需要通过以下命令安装HyperOpt:

```
$ pip install --upgrade git+git://github.com/hyperopt/hyperopt.git
```
该算法需要使用[HyperOpt搜索空间规范](https://github.com/hyperopt/hyperopt/wiki/FMin)。可以像下面这样使用HyperOptSearch:

```
tune.run(... , search_alg=HyperOptSearch(hyperopt_space, ... ))
```
一个例子可以在[hyperopt_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/hyperopt_example.py)中找到。

|[class ray.tune.suggest.hyperopt.HyperOptSearch(space, max_concurrent=10, reward_attr='episode_reward_mean', points_to_evaluate=None, **kwargs)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/hyperopt.html#HyperOptSearch)|  
|--|
基础：`ray.tune.suggest.suggestion.SuggestionAlgorithm`
HyperOpt的一个包装器提供试验建议。
需要从源代码安装HyperOpt。使用树结构的Parzen Estimators算法，尽管可以简单地扩展为支持HyperOpt使用的任何算法。HyperOpt不会跟踪外部添加的试验。


    参数：
    	space (dict) –HyperOpt配置。参数将从该配置中取样，并用于覆盖在变量生成过程中生成的参数。
    	max_concurrent (int) –最大同时试验次数。默认为10。
    	reward_attr (str) – 训练结果目标值属性。这指的是一个递增的值。
    	points_to_evaluate (list) –运行前初始参数suggestions。这适用于当您已经有了一些好的参数时，
    		您希望hyperopt首先运行，以帮助TPE算法为未来的参数提供更好的suggestions。需要是超选
    		项命名变量的dict列表。选择变量应该由列表中的索引表示(参见示例)
    	

例子：


    >>> space = {
    >>>     'width': hp.uniform('width', 0, 20),
    >>>     'height': hp.uniform('height', -100, 100),
    >>>     'activation': hp.choice("activation", ["relu", "tanh"])
    >>> }
    >>> current_best_params = [{
    >>>     'width': 10,
    >>>     'height': 0,
    >>>     'activation': 0, # The index of "relu"
    >>> }]
    >>> algo = HyperOptSearch(
    >>>     space, max_concurrent=4, reward_attr="neg_mean_loss",
    >>>     points_to_evaluate=current_best_params)

#### ４．SigOpt Search

SigOptSearch是一种由[SigOpt](https://sigopt.com/)支持的搜索算法，用于执行 **基于顺序模型的超参数优化** 。注意，这个类没有扩展`ray.tune.suggest.BasicVariantGenerator`。因此，在使用SigOptSearch时，将无法使用Tune的默认变量生成/搜索空间声明。

为了使用这个搜索算法，你需要通过以下命令安装SigOpt:

```
$ pip install sigopt
```

该算法要求用户拥有一个SigOpt API密钥来向API发出请求。将API令牌存储为名为SIGOPT_KEY的环境变量，如下所示：

```
$ export SIGOPT_KEY= ...
```

该算法需要使用SigOpt实验和空间规范。可以使用SigOptSearch如下:

```
tune.run(... , search_alg=SigOptSearch(sigopt_space, ... ))
```

一个例子可以在[sigopt_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/sigopt_example.py)中找到。

| [class ray.tune.suggest.sigopt.SigOptSearch(space, name='Default Tune Experiment', max_concurrent=1, reward_attr='episode_reward_mean', **kwargs)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/sigopt.html#SigOptSearch) |  
|--|

基础：`ray.tune.suggest.suggestion.SuggestionAlgorithm`
SigOpt的一个包装器提供试验建议。
需要安装SigOpt。要求用户将其SigOpt API密钥本地存储为SIGOPT_KEY上的环境变量。

```
参数：
	space (list of dict) – SigOpt配置。参数将从该配置中取样，并用于覆盖在变量生成过程中生成的参数。
	name (str) – 实验的名称。通过这个名称请求SigOpt。
	max_concurrent (int) –基于用户的SigOpt计划支持的最大并发试验数量。默认为1。
	reward_attr (str) –训练结果目标值属性。这指的是一个递增的值。
```
例子：

```
>>> space = [
>>>     {
>>>         'name': 'width',
>>>         'type': 'int',
>>>         'bounds': {
>>>             'min': 0,
>>>             'max': 20
>>>         },
>>>     },
>>>     {
>>>         'name': 'height',
>>>         'type': 'int',
>>>         'bounds': {
>>>             'min': -100,
>>>             'max': 100
>>>         },
>>>     },
>>> ]
>>> algo = SigOptSearch(
>>>     space, name="SigOpt Example Experiment",
>>>     max_concurrent=1, reward_attr="neg_mean_loss")
```

**注**：此算法需要密钥才可以使用，大款可以使用。
#### ５．Nevergrad Search
`NevergradSearch`是一个由[Nevergrad](https://github.com/facebookresearch/nevergrad)支持的搜索算法，用于执行基于顺序模型的超参数优化。注意，这个类没有扩展`ray.tune.suggest.BasicVariantGenerator`，因此，在使用NevergradSearch时，您将无法使用Tune的默认变量生成/搜索空间声明。

为了使用这个搜索算法，需要通过以下命令安装Nevergrad：

```
$ pip install nevergrad
```

nevergrad是一个Python 3.6的库。

该算法需要使用`nevergrad`提供的优化器，其中有许多选项。一个很好的纲要可以在他们的自述[优化](https://github.com/facebookresearch/nevergrad/blob/master/docs/optimization.md#Choosing-an-optimizer)部分找到。你可以像下面这样使用NevergradSearch:

```
tune.run(... , search_alg=NevergradSearch(optimizer, parameter_names, ... ))
```

tune的一个例子可以在[nevergrad_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/nevergrad_example.py)中找到。

|[class ray.tune.suggest.nevergrad.NevergradSearch(optimizer, parameter_names, max_concurrent=10, reward_attr='episode_reward_mean', **kwargs)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/nevergrad.html#NevergradSearch)|  
|--|
基础：`ray.tune.suggest.suggestion.SuggestionAlgorithm`
Nevergrad 的一个包装器提供试验建议。

要求安装Nevergrad。Nevergrad是一个来自Facebook的开源工具，用于参数和/或超参数的派生自由优化。它在标准的问与说接口中提供了广泛的优化器。

```
参数：
	optimizer (nevergrad.optimization.Optimizer)-优化器提供从Nevergrad。
	parameter_names (list) –参数名称列表。应该匹配优化器输出的维度。或者，如果优化器已经使用kwargs进行了
		检测，则将其设置为None(请参阅nevergrad v0.2.0)。
	max_concurrent (int) –最大同时试验次数。默认为10。
	reward_attr (str) –训练结果目标值属性。这指的是一个递增的值。
```
例子：

```
>>> from nevergrad.optimization import optimizerlib
>>> instrumentation = 1
>>> optimizer = optimizerlib.OnePlusOne(instrumentation, budget=100)
>>> algo = NevergradSearch(optimizer, ["lr"], max_concurrent=4,
>>>                        reward_attr="neg_mean_loss")
```
**注意：**  在evergrad v0.2.0 +中，可以对优化器进行检测。 例如，以下将指定从1到2搜索“lr”。

```
>>> from nevergrad.optimization import optimizerlib
>>> from nevergrad import instrumentation as inst
>>> lr = inst.var.Array(1).bounded(1, 2).asfloat()
>>> instrumentation = inst.Instrumentation(lr=lr)
>>> optimizer = optimizerlib.OnePlusOne(instrumentation, budget=100)
>>> algo = NevergradSearch(optimizer, None, max_concurrent=4,
>>>                        reward_attr="neg_mean_loss")
```
#### ６．Scikit-Optimize Search
`SkOptSearch`是一种由[Scikit-Optimize](https://scikit-optimize.github.io/) 支持的搜索算法，可以最小化复杂且有噪音的黑盒函数问题。用于执行基于顺序模型的超参数优化。注意，这个类没有扩展`ray.tune.suggest.BasicVariantGenerator`，因此，在使用SkOptSearch时，将无法使用Tune的默认变量生成/搜索空间声明。
为了使用这个搜索算法， 需要通过以下命令安装scikit - optimization:

```
$ pip install scikit-optimize
```
该算法需要使用[scikit - optimization ask and tell接口](https://scikit-optimize.github.io/notebooks/ask-and-tell.html)。这个接口需要使用scikit - optimization提供的[优化器](https://scikit-optimize.github.io/#skopt.Optimizer)。你可以使用SkOptSearch如下:

```
optimizer = Optimizer(dimension, ...)
tune.run(... , search_alg=SkOptSearch(optimizer, parameter_names, ... ))
```

一个例子可以在[skopt_example.py](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/skopt_example.py)中找到。

|[class ray.tune.suggest.skopt.SkOptSearch(optimizer, parameter_names, max_concurrent=10, reward_attr='episode_reward_mean', points_to_evaluate=None, evaluated_rewards=None, **kwargs)](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/skopt.html#SkOptSearch)  |
|--|
基础：`ray.tune.suggest.suggestion.SuggestionAlgorithm`
skopt 的一个包装器提供试验建议。

```
参数：
	optimizer (skopt.optimizer.Optimizer) –优化器提供从skopt。
	parameter_names (list) – 参数名称列表。应该匹配优化器输出的维度。
	max_concurrent (int) –最大同时试验次数。默认为10。
	reward_attr (str) –训练结果目标值属性。这指的是一个递增的值。
	points_to_evaluate (list of lists) –在从优化器中采样之前，首先要运行的点的列表，例如，这些可以是您
		已经知道的参数配置，可以很好地帮助优化器选择合适的值。每个点都是使用parameter_names给出
		的顺序定义的参数列表。
	evaluated_rewards (list) –如果您以前评估过作为points_to_evaluate传递进来的参数，那么您可以通过将
		奖励属性作为列表传递进来，从而避免重新运行那些试验，这样优化器就可以被告知结果，而不需要
		重新计算试验。必须与points_to_evaluate相同的长度。(参见调整/例子/ skopt_example.py)
```
例子：

```
>>> from skopt import Optimizer
>>> optimizer = Optimizer([(0,20),(-100,100)])
>>> current_best_params = [[10, 0], [15, -20]]
>>> algo = SkOptSearch(optimizer,
>>>     ["width", "height"],
>>>     max_concurrent=4,
>>>     reward_attr="neg_mean_loss",
>>>     points_to_evaluate=current_best_params)
```
#### 7.GeneticSearch 遗传搜索
GeneticSearch是在`ray.tune.automl`包内的自带搜索算法。

```
space = SearchSpace({
        ContinuousSpace('x1', 0, 4, 100),
        ContinuousSpace('x2', -2, 2, 100),
        ContinuousSpace('x3', 1, 5, 100),
        ContinuousSpace('x4', -3, 3, 100),
        DiscreteSpace('x5', [-1, 0, 1, 2, 3]),
    })
    algo = GeneticSearch(
        space,
        reward_attr="neg_mean_loss",
        max_generation= 10,
        population_size= 50)
```
根据上边代码进行参数和含义解释：

```
遗传优化
   
    search_space 样本空间
                ContinuousSpace('x1', 0, 4, 100)　把0到4分成100等分，即形成１００个数字，命名为x1 从中随机抽取
                DiscreteSpace('x5', [-1, 0, 1, 2, 3])　从-1,0,1,2,3　中随机抽取一个
    reward_attr　目标属性
    max_generation 是trail的迭代次数　
    population_size为初始的trail个数
    其中默认参数        population_decay=0.95,  基数衰减率
                      keep_top_ratio=0.2,     训练的最佳表现率　
                      selection_bound=0.4,    训练的选择率
                      crossover_bound=0.4     训练的交叉率
    也可以自己设置。
    总共进行max_generation次优化(此代码为１０次),每次trail个数为上一次的population_decay倍取整（此代码为０．９５）
    即：第一次运行50个训练，第二次会运行50*0.95 = 47个训练，第三次会运行47*0.95=44个训练 ...... 第１０次运行２８个训练
        共３８０个
```

#### 自定义搜索算法API
| [class ray.tune.suggest.SearchAlgorithm](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/search.html#SearchAlgorithm) |  
|--|

用于超参数搜索的事件处理程序API的接口。
和训练调度不同，搜索算法没有能力修改执行（停止和暂停试验）。

 - **add_configurations(experiments）：** 跟踪给定的实验说明。
	参数： experiments (Experiment | list | dict) – 实验运行。
	
 - **next_trials()：** 提供要排队进入试用运行器的试验对象。
	返回：返回试验列表。
	返回类型：trials (list)
	
 - **on_trial_result(trial_id, result)：** 调用由试验返回的每个中间结果。只有当试验处于运行状态时才会调用这个函数。
	参数：trial_id –试验标识符。
	
 - **on_trial_complete(trial_id, result=None, error=False, early_terminated=False)：**  完成试验的通知。
	参数：
		trial_id –试验标识符。
		result (dict) –默认为没有。当试验处于运行状态并自然完成或手动终止时，将向dict提供此通知。
		error (bool) – 默认值为False。如果试验处于运行状态且出现错误，则为真。
		early_terminated (bool) –默认值为False。如果试验在暂停或挂起状态下停止，则为真。

 - **is_finished()：** 如果没有剩余的试验排队进入TrialRunner，则返回True。可以在所有试验完成之前返回True。


#### 基于模型的建议的（Suggestion ）算法
通常，超参数搜索算法是基于模型的，并且可能非常容易实现。为此，可以扩展以下抽象类并实现`on_trial_result`、`on_trial_complete`和`_suggest`。抽象类将处理特定于Tune的样板文件，如创建试验和排队试验:

|[class ray.tune.suggest.SuggestionAlgorithm](https://ray.readthedocs.io/en/latest/_modules/ray/tune/suggest/suggestion.html#SuggestionAlgorithm)  |
|--|
基础：ray.tune.suggest.search.SearchAlgorithm
用于基于建议的算法的抽象类。
自定义搜索算法可以通过覆盖为试验提供生成参数的_`suggest`方法轻松扩展该类。
为了跟踪建议（suggestions ）及其相应的评估，方法`_suggest`将被传递一个`trial_id`，该id将在后续通知中使用。
例子：

```
>>> suggester = SuggestionAlgorithm()
>>> suggester.add_configurations({ ... })
>>> new_parameters = suggester._suggest()
>>> suggester.on_trial_complete(trial_id, result)
>>> better_parameters = suggester._suggest()
```

 - **_suggest(trial_id)：** 查询算法以检索下一组参数
	参数：trial_id –用于后续通知的试验ID。
	返回：如果存在，返回试验配置。否则，返回None，这将暂时停止TrialRunner的查询。
	返回类型：字典|空

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

