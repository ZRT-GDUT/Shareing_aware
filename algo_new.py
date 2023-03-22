import math
from typing import List, Dict

import device
import model_util


class Algo_new:

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

    def get_rsu_task_queue(self,
                           complete_tasks: dict):  # 将已经执行完毕的任务放到[[], [], [], [], [], [], [], [], [], []]，获得一个rsu完成任务的队列，execid的意思是job对应完成的rsu_id
        """

        :param complete_tasks: a dict, like--- {job-id: exec_rsu_id}
        :return:
        """
        rsu_task_queue = [[] for _ in range(self.rsu_num)]
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                job_id = task["job_id"]
                exec_id = complete_tasks[
                    job_id]  # 将已经执行完毕的任务放到[[], [], [], [], [], [], [], [], [], []]，获得一个rsu完成任务的队列，execid的意思是job对应完成的rsu_id
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

    def init_deployment_rsu_or_cloud(self, rsu_list, model_name_list, rsu_idx, model_idx):
        download_sub_models = []
        download_time_rsu = 0
        for model_offload_rsu_idx in rsu_list:  # 从其他存在task对应的model的RSU上下载model
            cpu_models = self.RSUs[model_offload_rsu_idx].get_cached_model(is_gpu=False)  # 获得其它RSU的model
            gpu_models = self.RSUs[model_offload_rsu_idx].get_cached_model(is_gpu=True)  # 获得其它RSU的model
            if not (model_name_list.isdisjoint(cpu_models) or model_name_list.isdisjoint(gpu_models)):
                # 遍历的RSU存在task对应的model
                inter_model = set()  # 存储RSU上存在的task所需要的model
                if not model_name_list.isdisjoint(cpu_models):
                    #  RSU的cpu存在对应的model
                    inter_model.add(cpu_models.intersection(model_name_list))
                if not model_name_list.isdisjoint(gpu_models):
                    #  RSU的gpu存在对应的model
                    inter_model.add(gpu_models.intersection(model_name_list))
                for model_download_name in inter_model:
                    model_idx_download, sub_model_idx = model_util.get_model_info(model_download_name)
                    download_sub_models.append(sub_model_idx)
                model_name_list = model_name_list - inter_model  # 除去task所需模型中已经下载好的模型
                download_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, download_sub_models,
                                                                          is_gpu=False)
                download_time_rsu = download_size / self.RSUs[model_offload_rsu_idx].rsu_rate + download_time_rsu
        if model_name_list:
            #  遍历所有rsu之后依旧没有task所需要的model，则从cloud上下载
            cloud_download_sub_models = []
            download_time_cloud = 0
            for model_download_name in model_name_list:
                model_idx_download, sub_model_idx = model_util.get_model_info(model_download_name)
                cloud_download_sub_models.append(sub_model_idx)
                download_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, cloud_download_sub_models,
                                                                          is_gpu=False)
                download_time_cloud = download_size / self.RSUs[rsu_idx].download_rate + download_time_cloud
        return download_time_rsu, download_time_cloud

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
            model_name_list = set()
            rsu_list = set(rsu_idx for rsu_idx in range(self.rsu_num))
            rsu_list.sort(key=lambda x: self.RSUs[x].rsu_rate)  # 根据RSU之间的通信速率进行排序
            for sub_model_idx in sub_models:  # 获取当前task所需要的模型
                model_name_list.add(model_util.get_model_name(model_idx, sub_model_idx))
            cpu_models_first = self.RSUs[rsu_idx].get_cached_model(is_gpu=False)  # 获取当前RSU的缓存的模型
            gpu_models_first = self.RSUs[rsu_idx].get_cached_model(is_gpu=True)  # 获取当前RSU的缓存的模型
            if model_name_list.isdisjoint(cpu_models_first) and model_name_list.isdisjoint(gpu_models_first):
                # 当前RSU上不存在task对应的model，则从其他RSU上下载
                download_time_rsu, download_time_cloud = self.init_deployment_rsu_or_cloud(rsu_list, model_name_list, rsu_idx, model_idx)
            else:
                # 当前RSU上存在task对应的model
                inter_rsu_has_model = set()  # 存储RSU存在的task所需要的model
                download_model = set()
                if not model_name_list.isdisjoint(cpu_models_first):
                    #  RSU的cpu存在对应的model
                    inter_rsu_has_model.add(cpu_models_first.intersection(model_name_list))
                if not model_name_list.isdisjoint(gpu_models_first):
                    #  RSU的gpu存在对应的model
                    inter_rsu_has_model.add(gpu_models_first.intersection(model_name_list))
                download_model = model_name_list - inter_rsu_has_model
                download_time_rsu, download_time_cloud = self.init_deployment_rsu_or_cloud(rsu_list, download_model, rsu_idx, model_idx)
            download_time = download_time_cloud + download_time_rsu
            _latency = []
            add_gpu_model = []
            for sub_model_idx in sub_models:
                device_name, tmp_latency = self.cal_greedy_cpu_gpu(rsu_idx, model_idx, sub_model_idx,
                                                                   seq_num[model_idx][sub_model_idx],
                                                                   loading_time=0)  # seq_num指代应该使用latency中的哪一项，device_name指的是cpu or gpu
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
            object_value = object_value + tmp_time
            # object_value = object_value + tmp_time + queue_latency
            # queue_latency = queue_latency + tmp_time
            for sub_model_idx in sub_models:
                seq_num[model_idx][sub_model_idx] += 1
            if model_idx not in cpu_add_models:
                cpu_add_models[model_idx] = set(sub_models)  # ？？129
            else:
                cpu_add_models[model_idx].union(sub_models)
            if model_idx not in gpu_add_models:
                gpu_add_models[model_idx] = set(add_gpu_model)
            else:
                gpu_add_models[model_idx].union(add_gpu_model)
        return object_value

    # ------------------------------------------------------------------------------
    #                IARR algorithm
    #   iarr: main function of this algorithm
    #   arr: called by iarr
    #   cal_queue_latency: 当RSU rsu_idx 上的任务队列为task_list时，greedy策略下的总时延是多少
    # ------------------------------------------------------------------------------

    def iarr(self, task_list: List[dict], shared=True, min_gap=0.1) -> float:
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
            throughput, object_value, _ = self.arr(T_max)
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
                                                                   seq_num[model_idx][sub_model_idx],
                                                                   loading_time=loading_time)
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

    def cal_greedy_cpu_gpu(self, rsu_idx, model_idx, sub_model_idx, seq_num: int, loading_time=0) -> [str,
                                                                                                      float]:  # 判断是在cpu上执行还是gpu
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

    def arr(self, T_max):  # 通过ita算法获取一个task分配方法
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
            if rsu_queue_effect[rsu_idx] >= T_max / self.rsu_num:  # 不太理解
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

    def add_tasks(self, rsu_idx, max_latency, task_list: List, seq_num: List[List[int]],
                  is_shared=True):  # 被ita调用，只是判断能不能添加，并不是真的添加
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
