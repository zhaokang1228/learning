import ray

ray.init()
import time

# def f1():
#     time.sleep(3)
#
# @ray.remote
# def f2():
#     time.sleep(3)
#
# # The following takes ten seconds.
# [f1() for _ in range(6)]

# The following takes one second (assuming the system has at least ten CPUs).
# ray.get([f2.remote() for _ in range(6)])



# result_ids = [ray.put(i) for i in range(10)]
# ray.get(result_ids)  # [0, 1, gai2, 3, 4, 5, 6, 7, 8, 9]
#
# print(ray.get(result_ids[5]) )


# @ray.remote(num_return_vals=3)
# def return_multiple():
#     return 1, gai2, 3
#
#
# a_id, b_id, c_id = return_multiple.remote()
# print(ray.get(a_id))

#
#
# import numpy as np
#
# @ray.remote
# def generate_data():
#     return np.random.normal(size=1000)
#
# @ray.remote
# def aggregate_data(x, y):
#     return x + y
#
# # 生成一些随机数据。这将启动100个任务，这些任务将在各个节点上调度。
# # 结果数据将分布在集群的各个节点中。
# data = [generate_data.remote() for _ in range(100)]
#
# print(ray.get(data))
#
# # 执行树缩减。
# while len(data) > 1:
#     data.append(aggregate_data.remote(data.pop(0), data.pop(0)))
#
# #获取结果
# ray.get(data)
#
# print(ray.get(data))
# print(len(ray.get(data)[0]))



# import numpy as np
#
#
# def generate_data():
#     return np.random.normal(size=1000)
#
# def aggregate_data(x, y):
#     return x + y
#
# # 生成一些随机数据。这将启动100个任务，这些任务将在各个节点上调度。
# # 结果数据将分布在集群的各个节点中。
# data = [generate_data() for _ in range(100)]
#
# print(data)
# print(len(data[0]))
#
# # 执行树缩减。
# while len(data) > 1:
#     data.append(aggregate_data(data.pop(0), data.pop(0)))
#
# print(data)
# print(len(data[0]))




#
# import asyncio
# import time
# import ray
# from ray.experimental import async_api
#
# @ray.remote
# def f():
#     time.sleep(10)
#     return {'key1': ['value']}
#
# # ray.init()
# future = async_api.as_future(f.remote())
# aa = asyncio.get_event_loop().run_until_complete(future)  # {'key1': ['value']}
# print(aa)
#


# # Execute f serially
# def f():
#   time.sleep(10)
#   return 1
#
# results = [f() for i in range(4)]
# print(results)



# # Execute f in parallel.
# @ray.remote
# def f():
#     time.sleep(10)
#     return 1
#
# # ray.init()
# results = ray.get([f.remote() for i in range(4)])
#
# print(results)


# import asyncio
# # Execute f asynchronously.
# async def f():
#     await asyncio.sleep(1)
#     return 1
#
# loop = asyncio.get_event_loop()
# tasks = [f() for i in range(4)]
#
# results = loop.run_until_complete(asyncio.gather(tasks))

# import asyncio
# import time
# from ray.experimental import async_api
# @ray.remote
# def f():
#     time.sleep(1)
#     return 1
#
# # ray.init()
# loop = asyncio.get_event_loop()
# tasks = [async_api.as_future(f.remote()) for i in range(4)]
# results = loop.run_until_complete(asyncio.gather(tasks))
#
# print(results)
#

# import asyncio
# import time
# import ray
# from ray.experimental import async_api
#
# @ray.remote
# def f():
#     time.sleep(10)
#     return {'key1': ['value']}
#
# # ray.init()
# future = async_api.as_future(f.remote())
# aa = asyncio.get_event_loop().run_until_complete(future)  # {'key1': ['value']}
# print(aa)
#


#
# import ray.experimental.signal as signal
#
# # Define a user signal.
# class UserSignal(signal.Signal):
#     def __init__(self, value):
#           self.value = value
#
#     def get_value(self):
#           return self.value
#
# @ray.remote
# def send_signal(value):
#     signal.send(UserSignal(value))
#     return
#
# signal_value = 'simple signal'
# object_id = send_signal.remote(signal_value)
# # Wait up to 10sec to receive a signal from the task. Note the task is
# # identified by the object_id it returns.
# result_list = signal.receive([object_id], timeout=10)
# # Print signal values. This should print "simple_signal".
# # Note that result_list[0] is the signal we expect from the task.
# # The signal is a tuple where the first element is the first object ID
# # returned by the task and the second element is the signal object.
# print(result_list[0][1].get_value())

# @ray.remote
# def task():
#     raise Exception('exception message')
#
# object_id = task.remote()
# try:
#     ray.get(object_id)
# except Exception as e:
#     pass
# finally:
#     result_list = signal.receive([object_id], timeout=10)
#     # Expected signal is 'ErrorSignal'.
#     assert type(result_list[0][1]) == signal.ErrorSignal
#     # Print the error.
#     print(result_list[0][1].get_error())



import ray.experimental.signal as signal

# Define a user signal.
class UserSignal(signal.Signal):
    def __init__(self, value):
          self.value = value

    def get_value(self):
          return self.value


@ray.remote
class ActorSendSignals(object):
    def send_signals(self, value, count):
        for i in range(count):
            signal.send(UserSignal(value + str(i)))

@ray.remote
class ActorGetAllSignals(object):
    def __init__(self, num_expected_signals, *source_ids):
        self.received_signals = []
        self.num_expected_signals = num_expected_signals
        self.source_ids = source_ids

    def register_handle(self, handle):
        self.this_actor = handle

    def get_signals(self):
        new_signals = signal.receive(self.source_ids, timeout=10)
        self.received_signals.extend(new_signals)
        print(len(self.received_signals))
        print(self.received_signals[0][1].get_value())
        if len(self.received_signals) < self.num_expected_signals:
            print(len(self.received_signals))
            self.this_actor.get_signals.remote()
            print("11111")

    def get_count(self):
        return len(self.received_signals)

# Create two actors to send signals.
a1 = ActorSendSignals.remote()
a2 = ActorSendSignals.remote()
signal_value = 'simple signal'
count = 5
# Each actor sends five signals.
a1.send_signals.remote(signal_value, count)
a2.send_signals.remote(signal_value, count)

# res = signal.receive([a1, a2],timeout=10)
# print(res[0][1].get_value())
# print(res)
# print(len(res))
# res2 = signal.receive([a1, a2],timeout=10)
# print(len(res2))
# Create an actor that waits for all five signals sent by each actor.
b = ActorGetAllSignals.remote(2 * count, *[a1, a2])
# Provide actor to its own handle, so it can recursively call itself
# to get all signals from a1, and a2, respectively. This enables the actor
# execute other methods if needed.
ray.get(b.register_handle.remote(b))
b.get_signals.remote()
# Print total number of signals. This should be gai2*count = 10.
print(ray.get(b.get_count.remote()))