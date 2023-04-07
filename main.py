import model_util
from algo_new import Algo_new
from data import google_data_util
import device
import random

#random.seed(1023)

default_task_file = "data/600.csv"

second_loop_num = 10

base_seed = 10086


def algo_print(algo_name, **kwargs):
    print("*" * 100)
    print(" " * 10, algo_name, end="\t")
    print(kwargs)
    print("*" * 100)


def tmp_record(data):
    with open("tmp.data.txt", "a+") as f:
        f.write("{}\n".format(data))


def record_file(x, data, description):
    with open("performance.txt", "a+") as f:
        f.write("{}\n".format(description))
        f.write("{}\n".format(x))
        f.write("{}\n".format(data))
        f.write("\n\n")


def out_line():
    import datetime
    with open("performance.txt", "a+") as f:
        f.write("-" * 100)
        f.write("\n")
        f.write("{}\n".format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')))
    with open("tmp.data.txt", "a+") as f:
        f.write("-" * 100)
        f.write("\n")
        f.write("-" * 100)
        f.write("\n")
        f.write("{}\n".format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')))


def gen_by_random(gpu_ratio, download_rate, max_storage, rsu_rate, rsu_num, is_random=True):
    if is_random:
        return [device.RSU(gpu_ratio=gpu_ratio,
                           download_rate=download_rate,
                           max_storage=max_storage,
                           rsu_rate=rsu_rate) for _ in range(rsu_num)]
    else:
        rsu_idx = [i for i in range(rsu_num)]
        random.shuffle(rsu_idx)  # 打乱rsu_idx的顺序
        rsu_idx = rsu_idx[:int(gpu_ratio * rsu_num)]
        RSUs = []
        for idx in range(rsu_num):  # 这是考虑不随机的情况？
            if idx in rsu_idx:
                ratio = 1
            else:
                ratio = 0
            RSUs.append(device.RSU(gpu_ratio=ratio, download_rate=download_rate, rsu_rate=rsu_rate))
        return RSUs


def run_algo(rsu_num=20,
             gpu_ratio=0.5,
             filename=default_task_file,
             max_latency=1.6,
             max_storage=1700,
             rsu_rate=120,
             download_rate=550,
             seed=666,
             desc=None):
    #random.seed(seed)
    RSUs = gen_by_random(gpu_ratio, download_rate, max_storage, rsu_rate, rsu_num, is_random=True)
    init_model_deploy = random.uniform(0, 1)
    rand_rsu_id = random.randint(0, rsu_num - 1)
    rand_model_index = random.randint(0, len(model_util.Model_name)-1)
    rand_sub_model_index = random.randint(0, model_util.Sub_model_num[rand_model_index]-1)
    if init_model_deploy <= 0.3:
        # 部署大模型
        for sub_model_idx in range(model_util.Sub_model_num[rand_model_index]):
            RSUs[rand_rsu_id].add_model(rand_model_index, sub_model_idx, is_gpu=False)  # 默认部署在cpu
    elif 0.3 < init_model_deploy <= 0.6:
        # 部署小模型
        RSUs[rand_rsu_id].add_model(rand_model_index, rand_sub_model_index, is_gpu=False)
    print(RSUs[rand_rsu_id].get_cached_model())
    print(rand_rsu_id)
    alg = Algo_new(RSUs)  # 在当前随机生成的RSU数量进行实验
    task_list = google_data_util.process_task(rsu_num, filename, max_latency=max_latency)
    result = []
    algo_print(desc, rsu_num=rsu_num, gpu_ratio=gpu_ratio, filename=filename, storage=max_storage, seed=seed)
    result.append(alg.iarr(task_list))
    tmp_record(result)
    return result


def rsu_num_change():
    """
    RSU数量变化曲线图
    """
    results = []
    x_list = []
    for rsu_num in range(10, 31, 5):
        res = []
        x_list.append(rsu_num)
        tmp_record("\nrsu_num_change, rsu_num: {}".format(rsu_num))
        for seed in range(base_seed, base_seed + second_loop_num, 1):  # seed是什么
            tmp = run_algo(rsu_num=rsu_num,
                           seed=seed,
                           desc="rsu_num: {}/second_loop: {}".format(rsu_num, second_loop_num - base_seed))
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        results.append(res)
        print(res)
    print(results)
    record_file(x_list, results, "RSU_num  ")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    out_line()
    rsu_num_change()
