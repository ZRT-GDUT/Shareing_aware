import os
import random

import pandas as pd
from tqdm import trange
from typing import List
import os

import model_util

google_file = "part-00000-of-00500.csv"

timestamp_col = 0
jobid_col = 2
taskid_col = 3
machine_id_col = 4
event_type = 5

submit_event = 0


def time2timestamp(val):
    return int(val * 1000000)


def get_edge_id(machine_id, rsu_num):
    if pd.isna(machine_id):
        return random.randint(0, rsu_num - 1)
    machine_id = int(machine_id)
    return machine_id % rsu_num


def out_event(filename=google_file, time_start=600, time_end=610, out_filename="data.csv"):
    """
    generate the result file from google_file, while it is in [time_start, time_end]
    :param filename:
    :param time_start:
    :param time_end:
    :param out_filename:
    :return:
    """
    flag = False
    if not os.path.exists(filename):
        filename = os.path.join("data", filename)
        flag = True
    df = pd.read_csv(filename)
    row_num = df.shape[0]
    res = []
    time_start_stamp = time2timestamp(time_start)
    time_end_stamp = time2timestamp(time_end)
    for row_idx in trange(row_num):
        if df.iloc[row_idx][timestamp_col] >= time_start_stamp and \
                df.iloc[row_idx][timestamp_col] < time_end_stamp and \
                df.iloc[row_idx][event_type] == submit_event:
            res.append([df.iloc[row_idx][timestamp_col]
                           , df.iloc[row_idx][jobid_col]
                           , df.iloc[row_idx][taskid_col]])
    df = pd.DataFrame(res, columns=["timestamp", "job_id", "task_id"])
    # if not flag:
    #     out_filename = os.path.join("data", out_filename)
    df.to_csv(out_filename)


def get_request_type(sub_model_num, max_k=4):  # 随机生成任务需要用到那些sub_model
    order = [i for i in range(sub_model_num)]
    k = random.randint(1, sub_model_num)
    if k > max_k:
        k = max_k
    res = random.sample(order, k)
    res.sort()
    return res


def process_task(rsu_num, filename, max_sub_task_num=10, max_latency=50) -> List[dict]:  # 任务里面有哪些属性
    df = pd.read_csv(filename)
    task_lists = []
    for col_idx in range(df.shape[0]):  # 遍历每一行数据
        info = {}
        info["job_id"] = col_idx
        info["rsu_id"] = random.randint(0, rsu_num - 1)
        info["model_idx"] = random.randint(0, len(model_util.Model_name) - 1)
        task_num = min([max_sub_task_num, model_util.Sub_model_num[info["model_idx"]]])
        info["sub_model"] = get_request_type(task_num)
        info["latency"] = random.uniform(max_latency * 0.8, max_latency)
        info["model_structure"] = list(model_util.get_model(info["model_idx"]).get_sub_module_by_model_idx_all(info["sub_model"]))
        task_lists.append(info)
    return task_lists


def outfiles():
    for start in range(600, 710, 10):
        time_end = start + 10
        out_event(time_start=start, time_end=time_end, out_filename="{}.csv".format(start))


if __name__ == '__main__':
    # res = process_task(10, "data.csv")
    # for _res in res:
    #     print(_res)
    outfiles()
