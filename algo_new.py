import math
from typing import List, Dict

import device
import model_util
import pulp as pl


class Algo_new:

    def __init__(self, RSUs: List[device.RSU]):
        self.RSUs = RSUs  # 6
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
        for rsu_idx in range(self.rsu_num):
            obj_value.append(self.call_obj(rsu_idx, rsu_task_queue[rsu_idx], is_shared=is_shared))
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

    def call_obj(self, rsu_idx, task_list: List, is_shared=True):  # 获得每个rsu的目标值
        object_value = 0
        cpu_add_models = {}
        gpu_add_models = {}
        i = 0
        for task in task_list:
            model_idx = task["model_idx"]
            sub_models = task["sub_model"]
            generated_id = task["rsu_id"]
            model = model_util.get_model(model_idx)  # model代表大模型
            download_size = self.RSUs[rsu_idx].cal_extra_caching_size(model_idx, sub_models)  # 为何设置为false
            download_time = download_size / self.RSUs[rsu_idx].download_rate
            _latency = []
            add_gpu_model = []
            for sub_model_idx in sub_models:
                device_name, tmp_latency = self.cal_greedy_cpu_gpu(rsu_idx, model_idx,
                                                                   sub_model_idx)  # seq_num指代应该使用latency中的哪一项，device_name指的是cpu or gpu
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
                self.RSUs[rsu_idx].seq_num[model_idx][sub_model_idx] += 1
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
        model_idx_jobid_list = self.generate_jobid_model_idx(task_list)
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
        print("T_max:", T_max)
        while T_max - T_min >= min_gap:
            throughput, object_value, _ = self.arr(T_max, model_idx_jobid_list, task_list, completed_tasks)
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

    def generate_jobid_model_idx(self, task_list: List[dict]):
        model_idx_jobid_list = []
        for task in task_list:
            model_idx_jobid_list.append(task["model_idx"])
        return model_idx_jobid_list

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

    def get_exec_latency(self, rsu_idx, model_idx, sub_model_idx):  # 获得在cpu和gpu上执行的latency
        model = model_util.get_model(model_idx)
        if self.RSUs[rsu_idx].has_gpu:
            gpu_idx = self.RSUs[rsu_idx].gpu_idx
            gpu_latency = model.cal_execution_delay(sub_model_idx, self.RSUs[rsu_idx].seq_num[model_idx][sub_model_idx],
                                                    gpu_idx)
            return "gpu", gpu_latency
        else:
            cpu_idx = self.RSUs[rsu_idx].cpu_idx
            cpu_latency = model.cal_execution_delay(sub_model_idx, self.RSUs[rsu_idx].seq_num[model_idx][sub_model_idx],
                                                    cpu_idx)
            return "cpu", cpu_latency

    def cal_greedy_cpu_gpu(self, rsu_idx, model_idx, sub_model_idx) -> [str,
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
        device, _latency = self.get_exec_latency(rsu_idx, model_idx, sub_model_idx)
        # if gpu_latency < math.inf:
        #     return "gpu", gpu_latency + loading_time
        # else:
        #     return "cpu", cpu_latency
        return device, _latency

    def arr(self, T_max, model_idx_jobid_list, task_list, complete_tasks):  # 通过arr算法获取一个task分配方法
        def cal_greedy_cpu_gpu_new(rsu_idx, model_idx, sub_models, is_shared=True):
            latency_list = []
            for sub_model_idx in sub_models:
                seq_num = self.RSUs[rsu_idx].seq_num[model_idx][sub_model_idx]
                device, _latency = self.get_exec_latency(rsu_idx, model_idx, sub_model_idx)
                latency_list.append(_latency)
                self.RSUs[rsu_idx].seq_num[model_idx][sub_model_idx] += 1
            if is_shared:
                latency = max(latency_list)
            else:
                latency = sum(latency_list)
            return latency

        rsu_list = [i for i in range(self.rsu_num)]
        rsu_list_structure = [i for i in range(self.rsu_num + 1)]  # index=rsu_num denotes cloud,最后一个代表cloud
        x_task_structure = [[0 for _ in range(len(model_util.Sub_Model_Structure_Size))] for _ in
                            range(self.get_all_task_num())]  # 表示每一个任务对应了哪些model structure
        x_rsu_tasktype = []
        x_rsu_tasktype_relax = []
        x_rsu_model_structure = []
        x_rsu_model_structure_relax = []
        x_rsu_to_rsu_model_structure = []
        x_rsu_to_rsu_model_structure_relax = []
        y_rsu_task = {}
        y_rsu_task_relax = {}
        x_structure_model = [
            [[0 for _ in range(len(model_util.Sub_Model_Structure_Size))] for _ in range(model_util.Sub_model_num[i])]
            for i in range(len(model_util.Model_name))]  # 用来定义每个model需要哪些structure
        rsu_task_queue = self.get_rsu_task_queue(complete_tasks)  # 获得[[], [], [], [], [], [], [], [], [], []]的任务队列形式
        rsu_seq_num = []  # rsu的模型队列
        for rsu_idx in range(self.rsu_num):
            rsu_seq_num.append([])
            for i in range(len(model_util.Model_name)):
                rsu_seq_num[rsu_idx].append([0 for _ in range(model_util.Sub_model_num[i])])
        for task in task_list:
            task_job_id = task["job_id"]
            task_structure = task["model_structure"]
            for task_structure_idx in task_structure:
                x_task_structure[task_job_id][task_structure_idx] = 1
        for model_idx in range(len(model_util.Model_name)):
            for sub_model_idx in range(model_util.Sub_model_num[model_idx]):
                model_structure = model_util.get_model(model_idx).require_sub_model_all[sub_model_idx]
                for model_structure_idx in model_structure:
                    x_structure_model[model_idx][sub_model_idx][model_structure_idx] = 1
        print('x_structure_model:', x_structure_model)
        print('x_task_structure:', x_task_structure)
        for job_id in range(self.get_all_task_num()):
            y_rsu_task[job_id] = -1
        y_rsu_task_relax = y_rsu_task
        for rsu_idx in range(self.rsu_num):
            x_rsu_tasktype.append([[0 for _ in range(model_util.Sub_model_num[i])] for i in
                                   range(len(model_util.Model_name))])  # rsu_id:task_type,X_i_e  [[],[],[],[]...,[]]
            x_rsu_model_structure.append(
                [0 for _ in range(len(model_util.Sub_Model_Structure_Size))])  # rsu_id:model_structure,α_i_l
        for rsu_idx in range(self.rsu_num + 1):  # 多一个cloud
            x_rsu_to_rsu_model_structure.append([
                [0 for _ in range(len(model_util.Sub_Model_Structure_Size))] for _ in range(self.rsu_num)])  # β_l_i'_i
        x_rsu_tasktype_relax = x_rsu_tasktype
        x_rsu_model_structure_relax = x_rsu_model_structure
        x_rsu_to_rsu_model_structure_relax = x_rsu_to_rsu_model_structure
        rand_sub_model_idx = []
        rand_model_structure_idx = []
        rand_rsu_id = -1
        rand_model_idx = -1
        for rsu_idx in rsu_list:
            models = self.RSUs[rsu_idx].get_cached_model()
            # gpu_models = self.RSUs[rsu_idx].get_cached_model(is_gpu=True)
            if models:  # 判断rsu是否部署了模型
                rand_rsu_id = rsu_idx
                for cpu_model in models:
                    model_idx, sub_model_idx = model_util.get_model_info(cpu_model)
                    rand_model_idx = model_idx
                    rand_sub_model_idx.append(sub_model_idx)
                    x_rsu_tasktype[rsu_idx][model_idx][sub_model_idx] = 1
                    x_rsu_tasktype_relax[rsu_idx][model_idx][sub_model_idx] = 1
                    model = model_util.get_model(model_idx)
                    for model_structure_idx in model.require_sub_model_all[sub_model_idx]:
                        rand_model_structure_idx.append(model_structure_idx)
                break
            # if gpu_models:
            #     for gpu_model in gpu_models:
            #         model_idx, sub_model_idx = model_util.get_model_info(cpu_model)
            #         x_rsu_tasktype[rsu_idx][model_idx][sub_model_idx] = 1
            #         x_rsu_tasktype_relax[rsu_idx][model_idx].remove(sub_model_idx)
            #         model = model_util.get_model(model_idx)
            #         for model_structure_idx in model.require_sub_model[sub_model_idx]:
            #             x_rsu_model_structure[rsu_idx][model_structure_idx] = 1
            #             x_rsu_model_structure_relax[rsu_idx][model_structure_idx] = None
            #             for other_rsu_idx in rsu_list_structure:
            #                 x_rsu_to_rsu_model_structure[other_rsu_idx][rsu_idx][model_structure_idx] = 0
            #                 x_rsu_to_rsu_model_structure_relax[other_rsu_idx][rsu_idx][model_structure_idx] = None
        max_system_throughput = pl.LpProblem("max_system_throughput", sense=pl.LpMaximize)  # 定义最大化吞吐率问题
        print(rand_model_structure_idx, 6)
        print(rand_sub_model_idx)
        print(rand_rsu_id, 6)
        print(rand_model_idx)
        x_i_e = {(i, m, s): pl.LpVariable('x_i_e_{0}_{1}_{2}'.format(i, m, s), lowBound=0, upBound=1,
                                          cat=pl.LpContinuous)
                 for i in range(self.rsu_num)
                 for m in range(len(model_util.Model_name))
                 for s in range(model_util.Sub_model_num[m])}
        for sub_model_idx in rand_sub_model_idx:
            x_i_e[rand_rsu_id, rand_model_idx, sub_model_idx].fixValue(1)

        x_i_l = {(i, l): pl.LpVariable('x_i_l_{0}_{1}'.format(i, l),
                                       lowBound=0, upBound=1, cat=pl.LpContinuous)
                 for i in range(self.rsu_num + 1)
                 for l in range(len(model_util.Sub_Model_Structure_Size))}
        for model_structure_idx in rand_model_structure_idx:
            var = x_i_l[(rand_rsu_id, model_structure_idx)]
            var.fixValue(1)

        x_i_i_l = {(i, j, l): pl.LpVariable('x_i_i_l_{0}_{1}_{2}'.format(i, j, l), lowBound=0, upBound=1,
                                            cat=pl.LpContinuous)
                   for i in range(self.rsu_num + 1)
                   for j in range(self.rsu_num)
                   for l in range(len(model_util.Sub_Model_Structure_Size))}
        for other_rsu_idx in range(self.rsu_num + 1):
            for model_structure_idx in rand_model_structure_idx:
                var = x_i_i_l[(other_rsu_idx, rand_rsu_id, model_structure_idx)]
                var.fixValue(0)

        y_i_jk = {(i, j): pl.LpVariable('y_i_jk_{0}_{1}'.format(i, j), lowBound=0, upBound=1, cat=pl.LpContinuous)
                  for i in range(self.rsu_num)
                  for j in range(self.get_all_task_num())}
        z_i_jk_l = {(i, j, t, l): pl.LpVariable('z_{0}_{1}_{2}_{3}'.format(i, j, t, l), lowBound=0, upBound=1,
                                                cat=pl.LpContinuous)
                    for i in range(self.rsu_num + 1)
                    for j in range(self.rsu_num)
                    for t in range(self.get_all_task_num())
                    for l in range(len(model_util.Sub_Model_Structure_Size))}
        max_system_throughput += (pl.lpSum((y_i_jk[rsu_idx_lp, job_id_lp] for rsu_idx_lp in range(self.rsu_num))
                                           for job_id_lp in range(self.get_all_task_num())))  # 目标函数
        for rsu_idx_lp in range(self.rsu_num + 1):
            for other_rsu_idx_lp in range(self.rsu_num):
                for job_id_lp in range(self.get_all_task_num()):
                    for model_structure_idx_lp in range(len(model_util.Sub_Model_Structure_Size)):
                        max_system_throughput += (
                                    z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, model_structure_idx_lp] <= y_i_jk[
                                other_rsu_idx_lp, job_id_lp])
                        max_system_throughput += (
                                    z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, model_structure_idx_lp] <=
                                    x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp])
                        max_system_throughput += (
                                    z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, model_structure_idx_lp] >=
                                    y_i_jk[other_rsu_idx_lp, job_id_lp] + x_i_i_l[
                                        rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] - 1)
        for job_id_lp in range(self.get_all_task_num()):
            max_system_throughput += (pl.lpSum(
                y_i_jk[rsu_idx_lp, job_id_lp] * cal_greedy_cpu_gpu_new(rsu_idx_lp, task_list[job_id_lp]["model_idx"],
                                                                       task_list[job_id_lp]["sub_model"]) for
                rsu_idx_lp in range(self.rsu_num)) +
                                      pl.lpSum((y_i_jk[other_rsu_idx_lp, job_id_lp] * model_util.get_model(
                                          model_idx_jobid_list[job_id_lp]).single_task_size / self.RSUs[
                                                    rsu_idx_lp].rsu_rate
                                                if task_list[job_id_lp][
                                                       "rsu_id"] == rsu_idx_lp and rsu_idx_lp != other_rsu_idx_lp else 0
                                                for rsu_idx_lp in range(self.rsu_num))
                                               for other_rsu_idx_lp in range(self.rsu_num)) +
                                      pl.lpSum(
                                          ((z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, model_structure_idx_lp] *
                                            x_task_structure[job_id_lp][model_structure_idx_lp] *
                                            model_util.Sub_Model_Structure_Size[model_structure_idx_lp]
                                            / (self.RSUs[rsu_idx_lp].rsu_rate if rsu_idx_lp != self.rsu_num else
                                               self.RSUs[other_rsu_idx_lp].download_rate)
                                            if other_rsu_idx_lp != rsu_idx_lp else 0
                                            for model_structure_idx_lp in
                                            range(len(model_util.Sub_Model_Structure_Size)))
                                           for other_rsu_idx_lp in range(self.rsu_num))
                                          for rsu_idx_lp in range(self.rsu_num + 1)) <= task_list[job_id_lp]["latency"])
        for rsu_idx_lp in range(self.rsu_num):
            for other_rsu_idx_lp in range(self.rsu_num):
                max_system_throughput += (pl.lpSum(
                    model_util.get_model(model_idx_jobid_list[job_id_lp]).single_task_size * y_i_jk[
                        other_rsu_idx_lp, job_id_lp] / self.RSUs[rsu_idx_lp].rsu_rate
                    if rsu_idx_lp != other_rsu_idx_lp and task_list[job_id_lp]["rsu_id"] == rsu_idx_lp else 0
                    for job_id_lp in range(self.get_all_task_num()))
                                          + pl.lpSum(x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] *
                                                     model_util.Sub_Model_Structure_Size[model_structure_idx_lp] /
                                                     self.RSUs[
                                                         rsu_idx_lp].rsu_rate if rsu_idx_lp != other_rsu_idx_lp else 0
                                                     for model_structure_idx_lp in range(
                            len(model_util.Sub_Model_Structure_Size))) <= T_max)  # Constraint(34)
        for rsu_idx_lp in range(self.rsu_num):
            max_system_throughput += (pl.lpSum(
                x_i_i_l[self.rsu_num, rsu_idx_lp, model_structure_idx_lp] * model_util.Sub_Model_Structure_Size[
                    model_structure_idx_lp]
                / self.RSUs[rsu_idx].download_rate for model_structure_idx_lp in
                range(len(model_util.Sub_Model_Structure_Size))) <= T_max)  # Constraint(35)
        for rsu_idx_lp in range(self.rsu_num):
            max_system_throughput += (pl.lpSum(
                y_i_jk[rsu_idx_lp, job_id_lp] * cal_greedy_cpu_gpu_new(rsu_idx_lp,
                                                                       task_list[job_id_lp]["model_idx"],
                                                                       task_list[job_id_lp]["sub_model"]) for
                job_id_lp in range(self.get_all_task_num())) <= T_max)  # Constraint(36)
        for job_id_lp in range(self.get_all_task_num()):
            max_system_throughput += (pl.lpSum(
                y_i_jk[rsu_idx_lp, job_id_lp] for rsu_idx_lp in range(self.rsu_num)) <= 1)  # Constraint(37)
        for rsu_idx_lp in range(self.rsu_num):
            max_system_throughput += (pl.lpSum((x_i_i_l[other_rsu_idx_lp, rsu_idx_lp, model_structure_idx_lp] *
                                                model_util.Sub_Model_Structure_Size[model_structure_idx_lp]
                                                for other_rsu_idx_lp in range(self.rsu_num + 1))
                                               for model_structure_idx_lp in
                                               range(len(model_util.Sub_Model_Structure_Size))) +
                                      pl.lpSum(y_i_jk[rsu_idx_lp, job_id_lp] * model_util.get_model(
                                          model_idx_jobid_list[job_id_lp]).single_task_size
                                               for job_id_lp in range(self.get_all_task_num())) <= self.RSUs[
                                          rsu_idx_lp].storage_capacity)  # Constraint(14)
        for rsu_idx_lp in range(self.rsu_num + 1):
            for other_rsu_idx_lp in range(self.rsu_num):
                for model_structure_idx_lp in range(len(model_util.Sub_Model_Structure_Size)):
                    max_system_throughput += (x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] <= x_i_l
                    [rsu_idx_lp, model_structure_idx_lp])  # Constraint(16)
        for rsu_idx_lp in range(self.rsu_num):
            for model_idx_lp in range(len(model_util.Model_name)):
                for sub_model_idx_lp in range(model_util.Sub_model_num[model_idx_lp]):
                    for model_structure_idx_lp in range(len(model_util.Sub_Model_Structure_Size)):
                        max_system_throughput += ((x_structure_model[model_idx_lp][sub_model_idx_lp][
                                                       model_structure_idx_lp] * x_i_e[
                                                       rsu_idx_lp, model_idx_lp, sub_model_idx_lp]) <=
                                                  pl.lpSum(
                                                      x_i_i_l[other_rsu_idx_lp, rsu_idx_lp, model_structure_idx_lp] for
                                                      other_rsu_idx_lp in range(self.rsu_num + 1)))  # Constraint(17)
        for rsu_idx_lp in range(self.rsu_num):
            for model_structure_idx_lp in range(len(model_util.Sub_Model_Structure_Size)):
                max_system_throughput += (pl.lpSum(
                    x_i_i_l[other_rsu_idx_lp, rsu_idx_lp, model_structure_idx_lp] for other_rsu_idx_lp in
                    range(self.rsu_num + 1)) <= 1)  # Constraint(18)
        for rsu_idx_lp in range(self.rsu_num):
            for job_id_lp in range(self.get_all_task_num()):
                for model_structure_idx_lp in range(len(model_util.Sub_Model_Structure_Size)):
                    max_system_throughput += (
                            y_i_jk[rsu_idx_lp, job_id_lp] * x_task_structure[job_id_lp][model_structure_idx_lp]
                            <= x_i_l[rsu_idx_lp, model_structure_idx_lp])  # Constraint(19)
        status = max_system_throughput.solve()
        print(pl.LpStatus[status])
        for v in y_i_jk.values():
            print(v.name, "=", v.varValue)
        for v in x_i_i_l.values():
            print(v.name, "=", v.varValue)
        print('objective =', pl.value(max_system_throughput.objective))
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
