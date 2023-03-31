"""
这个文件存放所有的算法代码，包括文章提出以及对比算法的代码
"""
import math
import random
import sys
from typing import List, Dict


import device
import model_util
from tqdm import tqdm
from scipy import optimize as op
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import warnings


class Algo:
    def __init__(self, RSUs: List[device.RSU]):
        self.RSUs = RSUs
        self.rsu_num = len(RSUs)

    def set_RSUs(self, RSUs: List[device.RSU]):
        self.RSUs = RSUs
        self.rsu_num = len(self.RSUs)

    def generate_tasks(self, task_list: List[dict], shared=True):
        for rsu_idx in range(self.rsu_num):
            self.RSUs[rsu_idx].task_list.clear()
        if shared:  # 没太理解这一块的含义
            for task in task_list:
                rsu_id = task["rsu_id"]
                self.RSUs[rsu_id].task_list.append(task)
        else:
            job_id = 0
            for task in task_list:
                rsu_id = task["rsu_id"]
                for sub_task in task["sub_model"]:
                    info = {
                        "job_id": job_id,
                        "init_job_id": task["job_id"],
                        "rsu_id": task["rsu_id"],
                        "model_idx": task["model_idx"],
                        "latency": task["latency"],
                        "sub_model": [sub_task]
                    }
                    self.RSUs[rsu_id].task_list.append(info)
                    job_id += 1

    def print_task_list(self):
        for rsu_idx in range(self.rsu_num):
            print("-" * 100)
            print("rsu_idx: ", rsu_idx)
            for task in self.RSUs[rsu_idx].task_list:
                print(task)

    # ------------------------------------------------------------------------------
    #        tools
    # ------------------------------------------------------------------------------

    def get_rsu_task_queue(self, complete_tasks: dict):  # 将已经执行完毕的任务放到[[], [], [], [], [], [], [], [], [], []]，获得一个rsu完成任务的队列，execid的意思是job对应完成的rsu_id
        """

        :param complete_tasks: a dict, like--- {job-id: exec_rsu_id}
        :return:
        """
        rsu_task_queue = [[] for _ in range(self.rsu_num)]
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                job_id = task["job_id"]
                exec_id = complete_tasks[job_id]  # 将已经执行完毕的任务放到[[], [], [], [], [], [], [], [], [], []]，获得一个rsu完成任务的队列，execid的意思是job对应完成的rsu_id
                if exec_id < len(rsu_task_queue):
                    rsu_task_queue[exec_id].append(task)
        return rsu_task_queue

    # ------------------------------------------------------------------------------
    #                call queue latency
    # ------------------------------------------------------------------------------

    def calculate_objective_value(self, complete_tasks, is_shared=False) -> list:  # 计算目标值，调用call_obj得到每个rsu的目标值
        """

        :param complete_tasks: a dict, like--- {job-id: exec_rsu_id}
        :return:
        """
        if len(complete_tasks.keys()) != self.get_all_task_num():
            # some task is not be executed......
            # a infeasible solution
            return [1000 for _ in range(self.rsu_num)]  # 没有理解为什么是1000
        rsu_task_queue = self.get_rsu_task_queue(complete_tasks)  # 获得[[], [], [], [], [], [], [], [], [], []]的任务队列形式
        obj_value = []
        rsu_seq_num = []  # rsu的模型队列
        for rsu_idx in range(self.rsu_num):
            rsu_seq_num.append([])
            for i in range(len(model_util.Model_name)):
                rsu_seq_num[rsu_idx].append([0 for _ in range(model_util.Sub_model_num[i])])  # 给每个rsu都添加子模型的存储空间 比如[
                # [0, 0, 0, 0]， [0, 0, 0]，[0, 0, 0, 0] ]
        for rsu_idx in range(self.rsu_num):
            obj_value.append(self.call_obj(rsu_idx, rsu_task_queue[rsu_idx], rsu_seq_num[rsu_idx], is_shared=is_shared))
        return obj_value

    def call_obj(self, rsu_idx, task_list: List, seq_num: List[List[int]], is_shared=True):  # 获得每个rsu的目标值
        object_value = 0
        #queue_latency = 0
        cpu_add_models = {}
        gpu_add_models = {}
        for task in task_list:
            model_idx = task["model_idx"]
            sub_models = task["sub_model"]
            generated_id = task["rsu_id"]
            model = model_util.get_model(model_idx)  # model代表大模型
            download_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, sub_models, is_gpu=False)  # 为何设置为false
            download_time = download_size / self.RSUs[rsu_idx].download_rate
            _latency = []
            add_gpu_model = []
            for sub_model_idx in sub_models:
                if sub_model_idx in add_gpu_model:
                    loading_time = 0
                else:
                    loading_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, [sub_model_idx], is_gpu=True)
                    # loading_time = loading_size / self.RSUs[rsu_idx].gpu_load_rate
                    loading_time = 0  # 为什么是0
                device_name, tmp_latency = self.cal_greedy_cpu_gpu(rsu_idx, model_idx, sub_model_idx,
                                                                   seq_num[model_idx][sub_model_idx],
                                                                   loading_time=loading_time)  # seq_num指代应该使用latency中的哪一项，device_name指的是cpu or gpu
                _latency.append(tmp_latency)
                if device_name == "gpu":
                    add_gpu_model.append(sub_model_idx)  # ？？
            # for communication
            if generated_id != rsu_idx:
                offloading_time = model.single_task_size / self.RSUs[generated_id].rsu_rate
            else:
                offloading_time = 0
            if is_shared:
                tmp_time = max(_latency)
            else:
                tmp_time = sum(_latency)
            tmp_time = tmp_time + offloading_time + download_time
            object_value = object_value + tmp_time + queue_latency
            queue_latency = queue_latency + tmp_time
            for sub_model_idx in sub_models:
                seq_num[model_idx][sub_model_idx] += 1  # 为什么每次加1
            if model_idx not in cpu_add_models:
                cpu_add_models[model_idx] = set(sub_models)  #？？129
            else:
                cpu_add_models[model_idx].union(sub_models)
            if model_idx not in gpu_add_models:
                gpu_add_models[model_idx] = set(add_gpu_model)
            else:
                gpu_add_models[model_idx].union(add_gpu_model)
        return object_value

    # ------------------------------------------------------------------------------
    #                TPA algorithm
    #   tpa: main function of this algorithm
    #   ita: called by tpa
    #   cal_queue_latency: 当RSU rsu_idx 上的任务队列为task_list时，greedy策略下的总时延是多少
    # ------------------------------------------------------------------------------

    def tpa(self, task_list: List[dict], shared=True, min_gap=0.1) -> float:
        self.generate_tasks(task_list, shared=shared)  # 给每个RSU分配任务
        completed_tasks = {}
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                job_id = task["job_id"]
                completed_tasks[job_id] = rsu_idx
        t = self.calculate_objective_value(completed_tasks, is_shared=False)
        T_max = sum(t)
        t = 0
        T_min = 0
        T_temp = (T_max - T_min) / 2
        T = T_temp
        obj = T_max
        total_task_num = self.get_all_task_num()
        last_t_max = T_max
        while T_max - T_min >= min_gap:
            throughput, object_value, _ = self.ita(T_max)
            # print("throughput: {}, total: {}".format(throughput, total_task_num))
            # print("T_max: {}, T_min: {}\n".format(T_max, T_min))
            last_t_min = T_min
            if throughput != total_task_num:
                T = (T_max + T_temp) / 2
                T_min = T_temp
                T_temp = T
                T_max = (last_t_max + T_max) / 2
            else:
                T = (T_temp + t) / 2
                last_t_max = T_max
                T_max = T_temp
                T_temp = T
                if object_value < obj:
                    obj = object_value
                continue
            if last_t_min == T_min:
                break
        return obj

    def cal_queue_latency(self, rsu_idx, task_list: List[Dict], is_shared=False):
        """
        return the total latency when the task_list executed by RSU rsu_idx
        :param rsu_idx:
        :param task_list:
        :param is_shared: true: different model can share the model result.
        :return:
        """
        latency = 0
        cpu_list, gpu_list = dict(), dict()
        seq_num = [[0 for _ in range(model_util.Sub_model_num[i])] for i in range(len(model_util.Model_name))]
        for task in task_list:
            model_idx = task["model_idx"]
            sub_models = task["sub_model"]
            _latency = []
            model = model_util.get_model(model_idx)
            for sub_model_idx in sub_models:
                model_name = model_util.get_model_name(model_idx, sub_model_idx)
                # from cloud to cpu
                extra_size_in_cpu = model.get_extra_model_new_size(cpu_list.get(model_idx, set()), sub_model_idx)
                download_time = extra_size_in_cpu / self.RSUs[rsu_idx].download_rate
                # from cpu to gpu
                extra_size_in_gpu = model.get_extra_model_new_size(gpu_list.get(model_idx, set()), sub_model_idx)
                # loading_time = extra_size_in_gpu / self.RSUs[rsu_idx].gpu_load_rate
                loading_time = 0
                device_name, tmp_latency = self.cal_greedy_cpu_gpu(rsu_idx, model_idx, sub_model_idx,
                                                                   seq_num[model_idx][sub_model_idx], loading_time=loading_time)
                if device_name == "cpu":
                    self.RSUs[rsu_idx].add_model(model_idx, sub_model_idx, is_gpu=False)
                else:
                    self.RSUs[rsu_idx].add_model(model_idx, sub_model_idx, is_gpu=True)
                seq_num[model_idx][sub_model_idx] += 1
                _latency.append(tmp_latency)
            if is_shared:
                latency += max(_latency)
            else:
                latency += sum(_latency)
        return latency

    def get_exec_latency(self, rsu_idx, model_idx, sub_model_idx, seq_num):  # 获得在cpu和gpu上执行的latency
        cpu_idx = self.RSUs[rsu_idx].cpu_idx
        gpu_idx = self.RSUs[rsu_idx].gpu_idx
        model = model_util.get_model(model_idx)
        cpu_latency = model.cal_execution_delay(sub_model_idx, seq_num, cpu_idx)
        gpu_latency = math.inf
        if gpu_idx != -1:
            gpu_latency = model.cal_execution_delay(sub_model_idx, seq_num, gpu_idx)
        return cpu_latency, gpu_latency

    def cal_greedy_cpu_gpu(self, rsu_idx, model_idx, sub_model_idx, seq_num: int, loading_time=0) -> [str, float]:  # 判断是在cpu上执行还是gpu
        """
        calculate through greedy method, for the task to be executed in cpu or gpu
        :param rsu_idx:
        :param model_idx:
        :param sub_model_idx:
        :param seq_num:
        :param loading_time:
        :return:
        """
        cpu_latency, gpu_latency = self.get_exec_latency(rsu_idx, model_idx, sub_model_idx, seq_num)
        # if gpu_latency < math.inf:
        #     return "gpu", gpu_latency + loading_time
        # else:
        #     return "cpu", cpu_latency
        if gpu_latency + loading_time < cpu_latency:
            return "gpu", gpu_latency + loading_time
        else:
            return "cpu", cpu_latency

    def ita(self, T_max):   # 通过ita算法获取一个task分配方法
        def arrange_task() -> dict:  # 整合成 model-sub_model: task
            tasks = {}  # key是model-submodel,value是对应相同key的model
            for rsu_idx in range(self.rsu_num):
                for task in self.RSUs[rsu_idx].task_list:
                    model_idx = task["model_idx"]
                    sub_models = [str(sub_idx) for sub_idx in task["sub_model"]]
                    key = "{}-{}".format(model_idx, ",".join(sub_models))
                    if key not in tasks.keys():
                        tasks[key] = [task]
                    else:
                        tasks[key].append(task)
            return tasks

        tasks = arrange_task()
        tasks_list = list(tasks.keys())
        tasks_list.sort()
        uncompleted_tasks = set(i for i in range(self.get_all_task_num()))
        rsu_visited = set(rsu_idx for rsu_idx in range(self.rsu_num))
        throughput = 0
        record_task_dict = {}
        seq_num_rsu = []  # 存储model 例如[[0,0,0],[0,0,0],[0,0,0]]
        rsu_queue_effect = [0 for _ in range(self.rsu_num)]  # 记录每个task在rsu上执行所需要等待的时间
        for rsu_idx in range(self.rsu_num):
            self.RSUs[rsu_idx].clear_cached_model()
            self.RSUs[rsu_idx].queue_latency = 0
            seq_num_rsu.append(
                [[0 for _ in range(model_util.Sub_model_num[i])] for i in range(len(model_util.Model_name))])
        while len(uncompleted_tasks) != 0 and len(rsu_visited) != 0:
            temp = 0
            x_temp = None
            task_visit = tasks_list.copy()
            for task_info in task_visit:  # 对每种类型的任务进行遍历
                visited_order = list(rsu_visited)
                visited_order.sort(key=lambda x: self.RSUs[x].queue_latency)
                for rsu_idx in visited_order:
                    # 先判断能不能加模型，不能加的，就下一个
                    complete_tasks, cpu_add_models, gpu_add_models, extra_size, extra_queue_latency, used_time = self.add_tasks(
                        rsu_idx, T_max - sum(rsu_queue_effect), tasks[task_info], seq_num_rsu[rsu_idx], is_shared=True)
                    if len(complete_tasks.keys()) > temp:
                        temp = len(complete_tasks.keys())
                        x_temp = [rsu_idx, complete_tasks, cpu_add_models, gpu_add_models, extra_size,
                                  extra_queue_latency, used_time]
                    elif len(complete_tasks.keys()) == temp and x_temp is not None:
                        if extra_queue_latency < x_temp[5]:
                            x_temp = [rsu_idx, complete_tasks, cpu_add_models, gpu_add_models, extra_size,
                                  extra_queue_latency, used_time]
                # if temp == 0:
                #     task_visit.remove(task_info)
            # caching model  Line 21 of algorithm 1
            if temp == 0:
                break
            rsu_idx = x_temp[0]
            complete_tasks = x_temp[1]
            cpu_add_models = x_temp[2]
            gpu_add_models = x_temp[3]
            extra_size = x_temp[4]
            extra_queue_latency = x_temp[5]
            used_time = x_temp[6]
            rsu_queue_effect[rsu_idx] += used_time
            # update cache model of RSU according to x_temp
            for model_idx in cpu_add_models.keys():
                self.RSUs[rsu_idx].add_all_sub_model(model_idx, list(cpu_add_models[model_idx]), is_gpu=False)
            for model_idx in gpu_add_models.keys():
                self.RSUs[rsu_idx].add_all_sub_model(model_idx, list(gpu_add_models[model_idx]), is_gpu=True)
            # update queue latency
            self.RSUs[rsu_idx].queue_latency += extra_queue_latency
            self.RSUs[rsu_idx].task_size += extra_size
            # update throughput
            throughput += len(complete_tasks.keys())
            complete_task_set = set(complete_tasks.keys())
            uncompleted_tasks = uncompleted_tasks - complete_task_set
            for key in complete_tasks.keys():
                record_task_dict[key] = rsu_idx
            for key in tasks.keys():
                tmp_visit = tasks[key].copy()
                for task in tmp_visit:
                    if task["job_id"] in complete_tasks:
                        tasks[key].remove(task)
                if len(tasks[key]) == 0:
                    del key
            if rsu_queue_effect[rsu_idx] >= T_max / self.rsu_num:  #不太理解
                rsu_visited.remove(rsu_idx)
            if len(uncompleted_tasks) == 0:
                break
        # print("rsu_queue_effect: {}".format(rsu_queue_effect))
        # print("task_num: {}, len of record_task_dict: {}, uncompleted_task: {}".format(self.get_all_task_num(),
        #                                                                                len(record_task_dict.keys()),
        #                                                                                uncompleted_tasks))
        t = self.calculate_objective_value(record_task_dict, is_shared=True)
        object_value = sum(t)
        return throughput, object_value, []

    def add_tasks(self, rsu_idx, max_latency, task_list: List, seq_num: List[List[int]], is_shared=True):  # 被ita调用，只是判断能不能添加，并不是真的添加
        complete_tasks = {}
        cpu_add_models = {}
        gpu_add_models = {}
        init_task_size = self.RSUs[rsu_idx].task_size
        init_queue_latency = self.RSUs[rsu_idx].queue_latency
        cpu_cached_model = self.RSUs[rsu_idx].get_cached_model(is_gpu=False)
        gpu_cached_model = self.RSUs[rsu_idx].get_cached_model(is_gpu=True)
        used_time = 0  # 记录任务的完成时间，对最终目标的影响
        for task in task_list:
            model_idx = task["model_idx"]
            job_idx = task["job_id"]
            sub_models = task["sub_model"]
            generated_id = task["rsu_id"]
            latency_requirement = task["latency"]
            model = model_util.get_model(model_idx)
            download_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, sub_models, is_gpu=False)
            add_success_model_cpu = self.RSUs[rsu_idx].add_all_sub_model(model_idx, sub_models, is_gpu=False)
            download_time = download_size / self.RSUs[rsu_idx].download_rate
            if download_size + model.single_task_size <= self.RSUs[rsu_idx].get_surplus_size():
                _latency = []
                add_gpu_model = []
                for sub_model_idx in sub_models:
                    if sub_model_idx in add_gpu_model:
                        loading_time = 0
                    else:
                        loading_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, [sub_model_idx],
                                                                                 is_gpu=True)
                        # loading_time = loading_size / self.RSUs[rsu_idx].gpu_load_rate
                        loading_time = 0
                    device_name, tmp_latency = self.cal_greedy_cpu_gpu(rsu_idx, model_idx, sub_model_idx,
                                                                       seq_num[model_idx][sub_model_idx],
                                                                       loading_time=loading_time)
                    _latency.append(tmp_latency)
                    if device_name == "gpu":
                        add_gpu_model.append(sub_model_idx)
                # for communication
                if generated_id != rsu_idx:
                    offloading_time = model.single_task_size / self.RSUs[generated_id].rsu_rate
                else:
                    offloading_time = 0
                if is_shared:
                    tmp_time = max(_latency) + offloading_time
                else:
                    tmp_time = sum(_latency) + offloading_time
                if tmp_time + self.RSUs[rsu_idx].queue_latency + download_time + used_time <= max_latency \
                        and tmp_time + self.RSUs[rsu_idx].queue_latency + download_time <= latency_requirement:
                    # task is executed in RSU rsu_idx
                    used_time = used_time + tmp_time + self.RSUs[rsu_idx].queue_latency + download_time
                    self.RSUs[rsu_idx].add_task(task_size=model.single_task_size, exec_latency=tmp_time + download_time)
                    self.RSUs[rsu_idx].add_all_sub_model(model_idx, add_gpu_model, is_gpu=True)
                    complete_tasks[job_idx] = rsu_idx
                    for sub_model_idx in sub_models:  # 为什么这里是+1
                        seq_num[model_idx][sub_model_idx] += 1
                    if model_idx not in cpu_add_models:
                        cpu_add_models[model_idx] = set(sub_models)
                    else:
                        cpu_add_models[model_idx].union(sub_models)
                    if model_idx not in gpu_add_models:
                        gpu_add_models[model_idx] = set(add_gpu_model)
                    else:
                        gpu_add_models[model_idx].union(add_gpu_model)
                else:
                    # task can not be execution, thus, remove add success model
                    for sub_model_idx in add_success_model_cpu:
                        self.RSUs[rsu_idx].remove_model(model_idx, sub_model_idx, is_gpu=False)
        self.RSUs[rsu_idx].rebuild_model_list(cpu_cached_model, gpu_cached_model)
        extra_task_size = self.RSUs[rsu_idx].task_size - init_task_size
        extra_queue_latency = self.RSUs[rsu_idx].queue_latency - init_queue_latency
        self.RSUs[rsu_idx].task_size = init_task_size
        self.RSUs[rsu_idx].queue_latency = init_queue_latency
        return complete_tasks, cpu_add_models, gpu_add_models, extra_task_size, extra_queue_latency, used_time

    def get_all_task_num(self) -> int:
        """
        :return: the total number of tasks in RSUs
        """
        task_num = 0
        for rsu_idx in range(self.rsu_num):
            task_num += len(self.RSUs[rsu_idx].task_list)
        return task_num

    # ------------------------------------------------------------------------------
    #                random rounding algorithm
    # ------------------------------------------------------------------------------

    def random_rounding(self, task_list: List[dict], shared=False):
        warnings.warn("该对比方法不再使用", DeprecationWarning)
        self.generate_tasks(task_list, shared=shared)
        strategy = self.greedy_cache()
        # print(strategy)
        # exit(0)
        res = self.linear_prog(is_shared=shared)
        complete_tasks = self.round_strategy(res)
        val = self.calculate_objective_value(complete_tasks, is_shared=shared)
        return sum(val)

    def round_strategy(self, res):
        def rounding(round_value):
            if random.random() <= round_value:
                return 1
            return 0

        complete_tasks = {}
        total_task_num = self.get_all_task_num()

        def get_idx(job_id, rsu_idx):
            return rsu_idx * total_task_num + job_id

        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                job_id = task["job_id"]
                model_idx = task["model_idx"]
                sub_models = task["sub_model"]
                complete_tasks[job_id] = -1
                for _rsu_idx in range(self.rsu_num):
                    if self.RSUs[_rsu_idx].can_executed(model_idx, sub_models):
                        if rounding(res.x[get_idx(job_id, _rsu_idx)]) == 1:
                            complete_tasks[job_id] = _rsu_idx
                            break
                if complete_tasks[job_id] == -1:
                    _rsu_idx = -1
                    x = 0
                    for tmp_rsu_idx in range(self.rsu_num):
                        if res.x[get_idx(job_id, tmp_rsu_idx)] > x:
                            _rsu_idx = tmp_rsu_idx
                            x = res.x[get_idx(job_id, tmp_rsu_idx)]
                    complete_tasks[job_id] = _rsu_idx
                    if complete_tasks[job_id] == -1:
                        complete_tasks[job_id] = random.randint(0, self.rsu_num)
        return complete_tasks

    def greedy_cache(self):
        """
        the first step of LR
        :return: obtain the greedy caching strategy (greedy in throughput)
        """

        def get_task_list() -> dict:
            task_list = {}
            for rsu_idx in range(self.rsu_num):
                for task in self.RSUs[rsu_idx].task_list:
                    model_idx = task["model_idx"]
                    sub_models = task["sub_model"]
                    model_name = model_util.get_model_name(model_idx, sub_models)
                    task_list[model_name] = task_list.get(model_name, 0) + 1
            return task_list

        task_list = get_task_list()
        used_task_size = [0 for _ in range(self.rsu_num)]
        cache_strategy = [{} for _ in range(self.rsu_num)]
        models = list(task_list.keys())
        models.sort(key=lambda x: task_list[x], reverse=True)
        for model in models:
            if task_list[model] == 0:
                continue
            model_idx, sub_models = model.split("-")
            model_idx = int(model_idx)
            sub_models = sub_models.replace("[", "").replace("]", "").split(", ")
            sub_models = [int(t) for t in sub_models]
            model_size = model_util.get_model(model_idx).single_task_size
            selected_rsu_idx = 0
            execute_count = 0
            for rsu_idx in range(self.rsu_num):
                extra_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, sub_models)
                size_for_model = self.RSUs[rsu_idx].storage_capacity - self.RSUs[rsu_idx].get_surplus_size()
                if size_for_model / self.RSUs[rsu_idx].storage_capacity >= 1 / 5:
                    # 这个防止所有的模型都加载到同一个RSU上。
                    continue
                surplus_size = self.RSUs[rsu_idx].get_surplus_size() - extra_size - used_task_size[rsu_idx]
                execution_task_num = int(surplus_size / model_size)
                if execution_task_num > task_list[model]:
                    execution_task_num = task_list[model]
                if execute_count < execution_task_num:
                    execute_count = execution_task_num
                    selected_rsu_idx = rsu_idx
                used_task_size[selected_rsu_idx] += (model_size * execute_count)
                self.RSUs[selected_rsu_idx].add_all_sub_model(model_idx, sub_models, is_gpu=False)
                task_list[model] -= execute_count
                if model_idx not in cache_strategy[selected_rsu_idx].keys():
                    cache_strategy[selected_rsu_idx][model_idx] = set(sub_models)
                else:
                    cache_strategy[selected_rsu_idx][model_idx] = cache_strategy[selected_rsu_idx][model_idx].union(sub_models)
                # if task_list[model] == 0:
                #     task_list.pop(model)
        # for model_idx in range(len(model_util.Model_name)):
        #     for sub_model_idx in range(model_util.Sub_model_num[model_idx]):
        #         model_name = model_util.get_model_name(model_idx, [sub_model_idx])
        #         selected_rsu_idx = 0
        #         execute_count = 0
        #         model_size = model_util.get_model(model_idx).single_task_size
        #         for rsu_idx in range(self.rsu_num):
        #             extra_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, [sub_model_idx])
        #             size_for_model = self.RSUs[rsu_idx].storage_capacity - self.RSUs[rsu_idx].get_surplus_size()
        #             if size_for_model / self.RSUs[rsu_idx].storage_capacity >= 1 / 3:
        #                 # 这个防止所有的模型都加载到同一个RSU上。
        #                 continue
        #             surplus_size = self.RSUs[rsu_idx].get_surplus_size() - extra_size - used_task_size[rsu_idx]
        #             execution_task_num = int(surplus_size / model_size)
        #             if execution_task_num > task_list[model_name]:
        #                 execution_task_num = task_list[model_name]
        #             if execute_count < execution_task_num:
        #                 execute_count = execute_count
        #                 selected_rsu_idx = rsu_idx
        #         used_task_size[selected_rsu_idx] += (model_size * execute_count)
        #         self.RSUs[selected_rsu_idx].add_model(model_idx, sub_model_idx)
        #         task_list[model_name] -= execute_count
        #         if model_idx not in cache_strategy[selected_rsu_idx].keys():
        #             cache_strategy[selected_rsu_idx][model_idx] = [sub_model_idx]
        #         else:
        #             cache_strategy[selected_rsu_idx][model_idx].append(sub_model_idx)
        #         if task_list[model_name] == 0:
        #             task_list.pop(model_name)
        return cache_strategy

    def cal_load_time(self):
        sizes = []
        for rsu_idx in range(self.rsu_num):
            sizes.append(self.RSUs[rsu_idx].storage_capacity - self.RSUs[rsu_idx].get_surplus_size())
        load_time = []
        for rsu_idx in range(self.rsu_num):
            load_time.append(sizes[rsu_idx] / self.RSUs[rsu_idx].download_rate)
        return load_time

    def get_surplus_size(self):
        sizes = []
        for rsu_idx in range(self.rsu_num):
            sizes.append(self.RSUs[rsu_idx].get_surplus_size())
        return sizes

    def linear_prog(self, is_shared=False) -> OptimizeResult:
        warnings.warn("该方法不再使用，队列的等待值不能固定，否则得出的结果性能不行", DeprecationWarning)
        total_task_num = self.get_all_task_num()

        def get_vector():
            return [0 for _ in range(self.rsu_num * total_task_num)]

        def get_idx(job_id, rsu_idx):
            return rsu_idx * total_task_num + job_id

        A_neq, B_neq, A_eq, B_eq = [], [], [], []
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                job_id = task["job_id"]
                model_idx = task["model_idx"]
                sub_models = task["sub_model"]
                latency = task["latency"]
                # for constraint 15
                vector = get_vector()
                for _rsu_idx in range(self.rsu_num):
                    vector[get_idx(job_id, _rsu_idx)] = 1
                A_eq.append(vector)
                B_eq.append(1)
                # for constraint 13
                vector = get_vector()
                for _rsu_idx in range(self.rsu_num):
                    exec_time = []
                    for sub_model_idx in sub_models:
                        _, tmp = self.cal_greedy_cpu_gpu(_rsu_idx, model_idx, sub_model_idx, 0, 10)
                        exec_time.append(tmp)
                    tmp = max(exec_time) if is_shared else sum(exec_time)
                    vector[get_idx(job_id, _rsu_idx)] = tmp
                A_neq.append(vector)
                B_neq.append(latency)
        # for constraint 14
        sizes = self.get_surplus_size()
        for rsu_idx in range(self.rsu_num):
            vector = get_vector()
            for _rsu_idx in range(self.rsu_num):
                for task in self.RSUs[_rsu_idx].task_list:
                    job_id = task["job_id"]
                    model_idx = task["model_idx"]
                    model = model_util.get_model(model_idx)
                    vector[get_idx(job_id, rsu_idx)] = model.single_task_size
            A_neq.append(vector)
            B_neq.append(sizes[rsu_idx])
        # for obj
        load_time = self.cal_load_time()
        obj_vector = get_vector()
        bounds = [(0, 1) for _ in range(total_task_num * self.rsu_num)]
        for rsu_idx in range(self.rsu_num):
            models = {}
            cache_models = self.RSUs[rsu_idx].get_add_models()
            for model in cache_models:
                model_idx, sub_model_idx = model_util.get_model_info(model)
                if models.get(model_idx, 0) == 0:
                    models[model_idx] = {sub_model_idx}
                else:
                    models[model_idx].add(sub_model_idx)
            for _rsu_idx in range(self.rsu_num):
                for task in self.RSUs[_rsu_idx].task_list:
                    job_id = task["job_id"]
                    model_idx = task["model_idx"]
                    sub_models = task["sub_model"]
                    stored_set = models.get(model_idx, set())
                    if len(stored_set) == len(stored_set.union(set(sub_models))):
                        exec_time = []
                        for sub_model_idx in sub_models:
                            _, tmp = self.cal_greedy_cpu_gpu(rsu_idx, model_idx, sub_model_idx, 0, loading_time=100)
                            exec_time.append(tmp)
                        tmp = max(exec_time) if is_shared else sum(exec_time)
                    else:
                        tmp = 100
                        # bounds[get_idx(job_id, rsu_idx)] = (0, 0)
                    obj_vector[get_idx(job_id, rsu_idx)] = tmp + load_time[rsu_idx]

        # Methods = ['highs-ipm', 'highs', 'interior-point', 'simplex', 'revised simplex', 'highs-ds']
        Methods = ['highs-ipm', 'highs', 'interior-point', 'simplex']
        res = None
        for method in Methods:
            res = op.linprog(obj_vector, A_ub=A_neq, b_ub=B_neq, A_eq=A_eq, b_eq=B_eq, bounds=bounds, method=method)
            if (res is not None) and res.success:
                break
        return res

    # ------------------------------------------------------------------------------
    #                DQN algorithm
    # ------------------------------------------------------------------------------

    def init_task_offloading(self):
        _tasks = {}
        upper_bound = self.get_all_task_num() / self.rsu_num * 3.5  # 这个上界指的是每个rsu最多可执行的task数量吗
        cached_num = 5
        size_for_task_computing = [0 for _ in range(self.rsu_num)]
        size = [self.RSUs[rsu_idx].get_surplus_size() for rsu_idx in range(self.rsu_num)]
        task_num = [0 for _ in range(self.rsu_num)]
        cache_models = {}
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                job_id = task["job_id"]
                _tasks[job_id] = -1
                model_idx = task["model_idx"]
                sub_models = task["sub_model"]
                model_name = "{}-{}".format(model_idx, sub_models)
                if model_name not in cache_models.keys():  # 如果task对应的model不在已缓存model中，则需要对该model进行部署
                    cache_models[model_name] = []
                    rsu_idx_order = [i for i in range(self.rsu_num)]
                    rsu_idx_order.sort(key=lambda x: size[x] - size_for_task_computing[x], reverse=True)
                    i = 0
                    tmp = 0
                    while i < len(rsu_idx_order):
                        _rsu_idx = rsu_idx_order[i]
                        add_success_models = self.RSUs[_rsu_idx].add_all_sub_model(model_idx, sub_models, is_gpu=False)
                        size[_rsu_idx] = self.RSUs[_rsu_idx].get_surplus_size()
                        if size[_rsu_idx] <= 0:
                            for sub_model_idx in add_success_models:
                                self.RSUs[_rsu_idx].remove_model(model_idx, sub_model_idx, is_gpu=False)
                            size[_rsu_idx] = self.RSUs[_rsu_idx].get_surplus_size()
                        else:
                            tmp += 1
                            cache_models[model_name].append(_rsu_idx)
                        if tmp >= cached_num:  # 意思是一种模型最多只能在五个rsu上面部署吗
                            break
                        i += 1
                task_size = model_util.get_model(model_idx).single_task_size
                visited_order = cache_models[model_name].copy()
                visited_order.sort(key=lambda x: task_num[x])
                for _rsu_idx in cache_models[model_name]:  # 对当前task进行部署，在部署了当前task的model的rsu进行遍历部署
                    if task_num[_rsu_idx] >= upper_bound:
                        continue
                    if task_size + size_for_task_computing[_rsu_idx] < size[_rsu_idx]: # 部署结束，跳出循环
                        _tasks[job_id] = _rsu_idx
                        task_num[_rsu_idx] += 1
                        size_for_task_computing[_rsu_idx] += task_size
                        break
                if _tasks[job_id] == -1:  # 考虑当前task经过上面的for循环仍未被部署的情况，且需要对此时部署的rsu添加对应的model
                    _idx = [i for i in range(self.rsu_num)]
                    _idx.sort(key=lambda x: size[x] - size_for_task_computing[x])
                    for __idx in _idx:
                        if task_num[__idx] >= upper_bound:
                            continue
                        add_success_models = self.RSUs[__idx].add_all_sub_model(model_idx, sub_models,
                                                                                is_gpu=False)
                        size[__idx] = self.RSUs[__idx].get_surplus_size()
                        if size[__idx] - size_for_task_computing[__idx] >= task_size:
                            _tasks[job_id] = __idx
                            task_num[__idx] += 1
                            size_for_task_computing[__idx] += task_size
                            break
                        else:
                            for sub_model_idx in add_success_models:
                                self.RSUs[__idx].remove_model(model_idx, sub_model_idx, is_gpu=False)
                            size[__idx] = self.RSUs[__idx].get_surplus_size()
        return _tasks

    def init_task_offloading_random(self):
        _tasks = {}
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                job_id = task["job_id"]
                _tasks[job_id] = random.randint(0, self.rsu_num-1)
        return _tasks

    def dqn(self, task_list: List[dict], shared=False, num_epoch=500, random_init=False):

        def employ_action(action_value, complete_tasks):
            # 更新策略
            # 0: 完成修改
            # 1: 不满足约束
            # 2: 不需要修改
            task_num = len(complete_tasks.keys())
            action_value = int(action_value)
            rsu_id = int(action_value / task_num)
            task_id = action_value % task_num
            # print("task_id: {}, rsu_id: {}".format(task_id, rsu_id))
            if complete_tasks[task_id] == rsu_id:
                return 2
            complete_tasks[task_id] = rsu_id
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].clear_cached_model()
            if self.is_satisfied_constraint(complete_tasks):
                return 0
            return 1

        def get_observation(complete_tasks, is_shared=False) -> list:  # 获取目标值
            """
            return the latency of RSU....
            :param complete_tasks:
            :param is_shared:
            :return:
            """
            return self.calculate_objective_value(complete_tasks, is_shared=is_shared)

        self.generate_tasks(task_list, shared=shared)
        num_state = self.rsu_num
        total_task_num = self.get_all_task_num()
        num_action = self.rsu_num * total_task_num + 1
        DRL = DQN.DQN(num_state, num_action)
        best_optimal = sys.maxsize
        train_base = 3.0
        train_bais = 30.0
        REWARDS = []
        LOSS = []
        OPT_RESULT = []
        for epoch in tqdm(range(num_epoch), desc="dqn"):
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].clear_cached_model()
            if random_init:
                complete_tasks = self.init_task_offloading_random()
            else:
                complete_tasks = self.init_task_offloading()
            observation = get_observation(complete_tasks, is_shared=shared)
            if sum(observation) < best_optimal:
                best_optimal = sum(observation)
            init_state = sum(observation)
            total_reward = 0
            for _ in range(500):
                action_value = DRL.choose_action(observation)  # 获取动作
                if action_value == num_state - 1:
                    # print("DRL think this state is the optimal, thus break..")
                    DRL.store_transition(observation, action_value, 0, observation)
                    break
                observation = get_observation(complete_tasks)  # 实施动作前的观察值
                # employ action .....
                flag = employ_action(action_value, complete_tasks)  # 实施动作
                if flag == 2:   # 通过模型获得结果与实际结果一致，不需要修改，本次循环跳过，环境不需要改变
                    continue
                observation_ = get_observation(complete_tasks)  # 获取实施动作之后的观察值(environment)，实施动作后的观察值
                reward = sum(observation_) - sum(observation)  # 实施action前后的reward差值
                total_reward += reward
                if flag == 1:
                    reward = -100000
                    DRL.store_transition(observation, action_value, reward, observation_)
                    break
                DRL.store_transition(observation, action_value, reward, observation_)
                if sum(observation_) < best_optimal:
                    best_optimal = sum(observation_)
                observation = observation_
            REWARDS.append(total_reward)
            OPT_RESULT.append(best_optimal)
            # print("objective_value: {}".format(best_optimal))
            if epoch >= train_bais and epoch % train_base == 0:
                # print("DRL is learning......")
                loss = DRL.learn()
                LOSS.append(float(loss))
            if epoch % 50 == 0:
                # print("\nepoch: {}, objective_value: {}".format(epoch, best_optimal))
                pass
        # plt.plot(LOSS)
        # plt.title("loss curve......")
        # plt.show()
        # plt.plot(OPT_RESULT)
        # plt.title("best_optimal")
        # plt.ylabel("objective, minimal is better.")
        # plt.show()
        with open("loss.txt", "w+") as f:
            f.write("reward: {}\n".format(REWARDS))
            f.write("loss: {}\n".format(LOSS))
        return best_optimal

    def is_satisfied_constraint(self, complete_tasks: dict, print_info=False):
        def satisfy_latency_constraint(rsu_task_queue):
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].clear_cached_model()
            for rsu_idx in range(self.rsu_num):
                rsu_task_queue[rsu_idx].sort(key=lambda x: x["latency"])
                queue_latency = 0
                seq_num = [[0 for _ in range(model_util.Sub_model_num[i])] for i in
                           range(len(model_util.Sub_model_num))]
                self.RSUs[rsu_idx].clear_cached_model()
                for task in rsu_task_queue[rsu_idx]:
                    model_idx = task["model_idx"]
                    sub_models = task["sub_model"]
                    self.RSUs[rsu_idx].add_all_sub_model(model_idx, sub_models)
                    download_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, sub_models, is_gpu=True)
                    download_time = download_size / self.RSUs[rsu_idx].download_rate
                    latency_requirement = task["latency"]
                    for sub_model_idx in sub_models:
                        loading_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, [sub_model_idx],
                                                                                 is_gpu=True)
                        # loading_time = loading_size / self.RSUs[rsu_idx].gpu_load_rate
                        loading_time = 0
                        device_name, tmp_l = self.cal_greedy_cpu_gpu(rsu_idx, model_idx, sub_model_idx,
                                                                     seq_num[model_idx][sub_model_idx],
                                                                     loading_time=loading_time)
                        self.RSUs[rsu_idx].add_all_sub_model(model_idx, sub_models, is_gpu=(device_name == "gpu"))
                        queue_latency = queue_latency + tmp_l + download_time
                        if queue_latency > latency_requirement:
                            return False
                        download_time = 0
            return True

        def satisfy_rsu_size_capacity(rsu_task_queue):
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].clear_cached_model()
                task_num = [0 for _ in model_util.Model_name]
                for task in rsu_task_queue[rsu_idx]:
                    model_idx = task["model_idx"]
                    sub_models = task["sub_model"]
                    task_num[model_idx] += 1
                    self.RSUs[rsu_idx].add_all_sub_model(model_idx, sub_models)
                task_size = 0
                for model_idx in range(len(task_num)):
                    model = model_util.get_model(model_idx)
                    task_size += model.single_task_size * task_num[model_idx]
                if not self.RSUs[rsu_idx].satisfy_caching_constraint(task_size=task_size):
                    return False
            return True

        rsu_task_queue = self.get_rsu_task_queue(complete_tasks)
        if not satisfy_rsu_size_capacity(rsu_task_queue):
            if print_info:
                print("violate size capacity")
            return False
        if not satisfy_latency_constraint(rsu_task_queue):
            if print_info:
                print("violate latency capacity....................")
            return False
        return True

    # ------------------------------------------------------------------------------
    #                coalition algorithm
    #                  2021-TITS
    #  Edge Caching and Computation Management for Real-Time Internet of Vehicles: An Online and Distributed Approach
    # ------------------------------------------------------------------------------

    def preference_coalition(self, task_list, shared=False):
        self.generate_tasks(task_list, shared=shared)
        complete_tasks = self.init_task_offloading()
        total_task_num = self.get_all_task_num()
        loop = 1
        while True:
            can_break = True
            utility = sum(self.calculate_objective_value(complete_tasks, is_shared=shared))
            for task_idx in tqdm(range(total_task_num), desc="preference_coalition, loop: {}".format(loop)):
                remarked_rsu_idx = complete_tasks[task_idx]
                for rsu_idx in range(self.rsu_num):
                    complete_tasks[task_idx] = rsu_idx
                    if self.is_satisfied_constraint(complete_tasks, print_info=False):
                        tmp_utility = sum(self.calculate_objective_value(complete_tasks, is_shared=shared))
                        if tmp_utility < utility:
                            utility = tmp_utility
                            remarked_rsu_idx = rsu_idx
                            can_break = False
                complete_tasks[task_idx] = remarked_rsu_idx
            if can_break or loop >= 40:
                break
            loop += 1
        return utility