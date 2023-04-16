"""
定义论文中的所有设备
目前文章只使用RSU。
"""
import random
from typing import List

import model_util


class RSU:
    def __init__(self, gpu_ratio=0.5, max_storage=1200, download_rate=None, rsu_rate=None):
        # transmission rate
        self.seq_num = [[0 for _ in range(model_util.Sub_model_num[i])]for i in range(len(model_util.Model_name))]
        if download_rate is None:
            self.download_rate = random.uniform(450, 550)  # Mbps
        else:
            self.download_rate = download_rate
        if rsu_rate is None:
            self.rsu_rate = random.uniform(80, 120)  # Mbps
        else:
            self.rsu_rate = rsu_rate
        # computation
        self.gpu_idx = -1   # -1, no-gpu, 0, 1: gpu_type_idx
        if random.uniform(0, 1) < gpu_ratio:  #
            self.gpu_idx = get_device_id(random.randint(0, 1), is_gpu=True)
            self.has_gpu = True
            self.has_cpu = False
        else:
            self.cpu_idx = get_device_id(random.randint(0, 1), is_gpu=False)
            self.has_gpu = False
            self.has_cpu = True
        self.trans_cpu_gpu = 16 * 1024  # Gbps
        # storage
        self.storage_capacity = random.uniform(300, max_storage)  # to do ...
        self.gpu_load_rate = 1000
        # task
        self.task_list = []
        self.__caching_model_list = set()  # data: get_model_name(model_idx, sub_model_idx)
        self.__caching_model_list_gpu = set()
        self.queue_latency = 0
        self.task_size = 0

    def add_task(self, task_size, exec_latency):  # 计算add一个task之后的queue_latency，以及rsu存储的task size
        self.queue_latency += exec_latency
        self.task_size += task_size

    def satisfy_caching_constraint(self, task_size=0):
        return self.storage_capacity > self.cal_caching_size() + task_size

    def get_surplus_size(self):  # 获得rsu的剩余存储空间
        """
        get the size for task execution
        :return:
        """
        return self.storage_capacity - self.cal_caching_size(is_gpu=False) - self.cal_caching_size(is_gpu=True)

    def cal_caching_size(self):
        """
        :return: the caching size of inference in CPU or GPU.
        """
        cached_model = self.get_cached_model()
        return self.__cal_cache_size(cached_model)

    def get_cached_model(self) -> set:
        """
        get the cached model in CPU (default) or GPU
        :param is_gpu:
        :return:
        """
        if self.has_gpu:
            return self.__caching_model_list_gpu.copy()
        else:
            return self.__caching_model_list.copy()

    def __cal_cache_size(self, cache_models: set):  # 计算cache_models的size
        models = {}
        for model in cache_models:
            model_idx, sub_model_idx = model_util.get_model_info(model)
            if models.get(model_idx, 0) == 0:
                models[model_idx] = {sub_model_idx}
            else:
                models[model_idx].add(sub_model_idx)
        model_size = 0
        for model_idx in models.keys():
            model_idxs = models[model_idx]  # 每个model_idx对应的sub_model
            model = model_util.get_model(model_idx)
            model_size += model.require_model_size(model_idxs, is_share=True)
        return model_size

    def cal_extra_caching_size(self, model_idx, sub_models: List[int]):
        """
        calculate the cache size when model_idx[sub_models] are added.
        :param model_idx:
        :param sub_models:
        :param is_gpu:
        :return:
        """
        pre_model_size = self.cal_caching_size()  # 获得已缓存模型的size
        models = self.get_cached_model() # 获取已缓存模型
        for sub_model_idx in sub_models:
            model_name = model_util.get_model_name(model_idx, sub_model_idx)
            models.add(model_name)
        after_model_size = self.__cal_cache_size(models)
        return after_model_size - pre_model_size

    def add_all_sub_model(self, model_idx, sub_models: List[int], is_gpu=False) -> List[int]:
        """
        :param model_idx:
        :param sub_models:
        :param is_gpu:
        :return: the new added sub_model_idx
        """
        add_success_models = []
        for sub_model_idx in sub_models:
            if self.add_model(model_idx, sub_model_idx, is_gpu=is_gpu):
                add_success_models.append(sub_model_idx)
        return add_success_models

    def add_model(self, model_idx, sub_model_idx):
        """
        :param model_idx:
        :param sub_model_idx:
        :param is_gpu:
        :return: true-> add a new model, false-> model has been added.
        """
        model_name = model_util.get_model_name(model_idx, sub_model_idx)
        if self.has_gpu:
            size = len(self.__caching_model_list_gpu)
            self.__caching_model_list_gpu.add(model_name)  # 没有理解，意思是gpu有了，cpu有一样的model就要删除吗
            # if self.has_model(model_idx, sub_model_idx):
            #     self.remove_model(model_idx, sub_model_idx)
            return len(self.__caching_model_list_gpu) - size != 0
        else:
            size = len(self.__caching_model_list)  # 为什么这里是获得gpu的model数量
            self.__caching_model_list.add(model_name)
            # if self.has_model(model_idx, sub_model_idx):
            #     self.remove_model(model_idx, sub_model_idx)
            return len(self.__caching_model_list) - size != 0

    def has_model(self, model_idx, sub_model_idx):
        model_name = model_util.get_model_name(model_idx, sub_model_idx)
        if self.has_gpu:
            return model_name in self.__caching_model_list_gpu
        else:
            return model_name in self.__caching_model_list

    def remove_add_models(self, model_idx: int, sub_models: List[int], is_gpu=False):
        """
        remove model_idx-[sub_models] in RSU
        :param model_idx:
        :param sub_models:
        :return:
        """
        for sub_model_idx in sub_models:
            self.remove_model(model_idx, sub_model_idx, is_gpu=is_gpu)

    def remove_model(self, model_idx, sub_model_idx):
        """
        remove model_idx-sub_model_idx in RSU
        :param model_idx:
        :param sub_model_idx:
        :return:
        """
        model_name = model_util.get_model_name(model_idx, sub_model_idx)
        if self.has_gpu:
            self.__caching_model_list_gpu.remove(model_name)
        else:
            self.__caching_model_list.remove(model_name)

    def has_all_model(self, sub_models) -> bool:
        """
        :return the result whether RSU caches all sub_models....
        :param sub_models:
        :return:
        """
        return len(sub_models - self.__caching_model_list - self.__caching_model_list_gpu) == 0

    def can_executed(self, model_idx, sub_models):
        """
        return the result whether the task can be executed by the RSU
        :param model_idx:
        :param sub_models:
        :return:
        """
        s = set()
        for sub_model_idx in sub_models:
            s.add(model_util.get_model_name(model_idx, sub_model_idx))
        return self.has_all_model(s)

    def get_add_models(self):
        return self.get_cached_model(is_gpu=True).union(self.get_cached_model(is_gpu=False))

    def get_model_idx_series(self, model_idx, is_gpu=False) -> set:
        """
        get the model_idx series...
        if model_idx = 1, and the cached_model is '1-1', '0-2', '0-3', '1-3',
        then the result is set(1, 3).
        :param model_idx: model_series_idx
        :param is_gpu:  in cpu or gpu
        :return:
        """
        sub_model_idxs = set()
        if is_gpu:
            cached_model = self.__caching_model_list_gpu
        else:
            cached_model = self.__caching_model_list
        for model_info in cached_model:
            _model_idx, sub_model_idx = model_util.get_model_info(model_info)
            if model_idx == _model_idx:
                sub_model_idxs.add(sub_model_idx)
        return sub_model_idxs

    def clear_cached_model(self):
        """
        clear all cached model in RSU
        """
        self.__caching_model_list.clear()
        self.__caching_model_list_gpu.clear()

    def rebuild_model_list(self, cpu_list, gpu_list):
        self.__caching_model_list = cpu_list
        self.__caching_model_list_gpu = gpu_list


def get_device_id(device_id, is_gpu=False, gpu_num=2): # 没理解这个函数的作用
    if is_gpu:
        return device_id
    else:
        return device_id + gpu_num