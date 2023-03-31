"""
存放不同模型的一些数据
这个文献的实验用的
"""
import math
from typing import List, Set

Model_name = ["HuangModel", "ChenModel", "Georgescu"]
Sub_model_num = [4, 3, 4]
Sub_Model_Structure = [5, 5, 6]
Sub_Model_Structure_Size = [179.2257156, 49.20232773, 0.00195694, 49.20232773, 206.0307236,
                            161.1276855, 47.32210541, 2.297855377, 2.427921295, 3.480670929,
                            1.591918945, 1.591918945, 0.071289063, 0.071289063, 0.143920898, 0.071289063]


def get_model_name(model_idx, sub_model_idx):
    return "{}-{}".format(model_idx, sub_model_idx)

def get_model_structure(model_idx, model_structure_idx):
    model = get_model(model_idx)
    return model.sub_model_size(model_structure_idx)

def get_model_info(model_info: str):
    info = model_info.split("-")
    return int(info[0]), int(info[1])


def get_image_size(width, height, channel):
    """
    get the size of one image with width * height * channel
    :param width:
    :param height:
    :param channel:
    :return: size in Bytes
    """
    return width * height * channel / 8


def Bytes2Mb(size):
    return size / (1024 * 1024)


class BaseMethod:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(BaseMethod, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.tasks = []  # sub task list of different task
        self.latency = []  # latency of sub task list
        self.model_name = []  # the provided service type
        self.require_sub_model = []  # a model contains different sub model
        self.sub_model_size = []  # in Mb, the size of different sub model
        self.single_task_size = math.inf

    def cal_execution_delay(self, sub_model_idx, seq_num, device_id):
        """
        calculate the execution delay of one task, with service sub_model_idx
        :param sub_model_idx: the service of this model series
        :param seq_num: different seq_num will result in different execution latency
        :param device_id: executed in which device
        :return:
        """
        if sub_model_idx < 0 or sub_model_idx >= len(self.model_name):
            print("model idx is not legal.")
            return None
        seq = self.tasks[sub_model_idx]
        execution_delay = 0
        for sub_model in seq:
            if seq_num >= len(self.latency[sub_model][device_id]):  # 如果seq_num大于10，则按照最后一个值的latency执行
                execution_delay += self.latency[sub_model][device_id][-1]
            else:
                execution_delay += self.latency[sub_model][device_id][seq_num]
        return execution_delay

    def require_model_size(self, model_idxs, is_share=True):
        if is_share:
            sub_modules = self.get_sub_module_by_model_idx(model_idxs)
            return self.get_total_module_size(sub_modules)
        else:
            return self.__get_model_size(model_idxs)

    def get_sub_module_by_model_idx(self, model_idxs: Set[int]):
        sub_model_idx = set()
        for model_idx in model_idxs:
            models = self.require_sub_model[model_idx]
            for idx in models:
                sub_model_idx.add(idx)
        return sub_model_idx

    def get_sub_module_by_model_idx_all(self, model_idxs: Set[int]):
        sub_model_idx = set()
        for model_idx in model_idxs:
            models = self.require_sub_model_all[model_idx]
            for idx in models:
                sub_model_idx.add(idx)
        return sub_model_idx

    def get_extra_model_new_size(self, cached_model: Set[int], new_sub_model_idx: int):
        new_cached_model = cached_model.copy()
        new_cached_model.add(new_sub_model_idx)
        return self.get_extra_model_size(cached_model, new_cached_model)

    def get_extra_model_size(self, cached_model: Set[int], new_model: Set[int]):
        """
        calculate the extra model size.
        :param cached_model: [int]
        :param new_model:
        :return:
        """
        exist_module = self.get_sub_module_by_model_idx(cached_model)
        new_module = self.get_sub_module_by_model_idx(new_model)
        required_module = new_module - exist_module
        return self.get_total_module_size(required_module)

    def __get_model_size(self, model_idxs: List[int]):
        model_size = 0
        for model_idx in model_idxs:
            for sub_model in self.require_sub_model[model_idx]:
                model_size += self.sub_model_size[sub_model]
        return model_size

    def get_total_module_size(self, sub_modules: Set[int]):
        size = 0
        for idx in sub_modules:
            size += self.sub_model_size[idx]
        return size


class HuangModel(BaseMethod):
    def __init__(self):
        super(HuangModel, self).__init__()
        self.model_name = ["age estimation", "identity classfication", "age regression and age classification", "FAS"]
        self.tasks = [[0], [1], [2], [3]]
        self.sub_model_size = [179.2257156, 49.20232773, 0.00195694, 49.20232773, 206.0307236]  # 每一层的大小
        self.require_sub_model = [[0, 1], [0, 2], [0, 3], [0, 4]]  # 需要哪些层
        self.require_sub_model_all = [[0, 1], [0, 2], [0, 3], [0, 4]]  # 需要哪些层
        self.single_task_size = Bytes2Mb(get_image_size(112, 112, 3))
        self.latency = [
             #  每一个元素中有四项分别代表着使用哪一个cpu或gpu
            [  # 0
                [0.1973605, 0.1982527, 0.1977889, 0.2009442, 0.2027571, 0.204893, 0.1968997, 0.2066418, 0.1978363, 0.1971634],
                [0.2299988, 0.2331354, 0.2455341, 0.2345285, 0.2314658, 0.2365028, 0.2595511, 0.228322, 0.236145, 0.2300552],
                [0.1645303, 0.0192916, 0.0140529, 0.0140292, 0.0140198, 0.0140286, 0.0141624, 0.0140435, 0.0140469, 0.0140267],
                [0.1645303, 0.0192916, 0.0140529, 0.0140292, 0.0140198, 0.0140286, 0.0141624, 0.0140435, 0.0140469, 0.0140267]
            ],
            [  # 1
                [0.20755, 0.1955544, 0.191715, 0.2003362, 0.1924398, 0.1957741, 0.1987331, 0.1997689, 0.2000075, 0.1947023],
                [0.2192278, 0.2236783, 0.239304, 0.2314798, 0.2267759, 0.2299196, 0.2502516, 0.2345959, 0.2361438, 0.2236564],
                [0.197561, 0.0219852, 0.0196325, 0.0196459, 0.0190636, 0.018966, 0.0189843, 0.0188254, 0.018835, 0.0187074],
                [0.1770275, 0.0171917, 0.0130068, 0.0132736, 0.0132129, 0.0132522, 0.0131701, 0.0133126, 0.0130964, 0.012643]
            ],
            [  # 2
                [0.2110046, 0.206003, 0.2037366, 0.2111136, 0.2041781, 0.207264, 0.2106943, 0.2120338, 0.2119275, 0.205587],
                [0.22227, 0.2344547, 0.2515713, 0.2433699, 0.2384606, 0.2423721, 0.2629923, 0.246728, 0.2486427, 0.2352069],
                [0.1973163, 0.0200291, 0.0155259, 0.0154012, 0.0153429, 0.0153581, 0.0154425, 0.0152935, 0.0152867, 0.0149271],
                [0.1767534, 0.0174833, 0.0132263, 0.013547, 0.0134845, 0.0135205, 0.0134498, 0.0135853, 0.0133631, 0.0128731]
            ],
            [  # 3
                [0.39929, 0.3936066, 0.4082943, 0.417456, 0.3996705, 0.4069022, 0.4039221, 0.4075282, 0.4075422, 0.4049913],
                [0.4576806, 0.482701, 0.4764581, 0.4748913, 0.4733252, 0.4858324, 0.4780169, 0.478016, 0.4717729, 0.4905155],
                [0.1855786, 0.0416556, 0.039362, 0.041933, 0.041193, 0.0405013, 0.0407579, 0.0397856, 0.0399682, 0.0396941],
                [0.1545, 0.0244677, 0.0191376, 0.0187591, 0.0188785, 0.018762, 0.0188769, 0.0188578, 0.0188102, 0.0187884]
            ]
        ]


class ChenModel(BaseMethod):
    def __init__(self):
        super().__init__()
        self.model_name = ["SC", "SE", "SR"]
        self.tasks = [[0], [1], [2]]
        self.require_sub_model = [[0, 2], [0, 1, 3], [0, 1, 4]]
        self.require_sub_model_all = [[5, 7], [5, 6, 8], [5, 6, 9]]
        self.sub_model_size = [161.1276855, 47.32210541, 2.297855377, 2.427921295, 3.480670929]
        self.single_task_size = Bytes2Mb(get_image_size(416, 416, 3))
        self.latency = [
            [
                [1.345184, 1.3951448, 1.4360982, 1.4330647, 1.4047035, 1.4245633, 1.4154638, 1.4279191, 1.4171983, 1.4255445],
                [1.8276929, 1.9026842, 1.9229911, 1.908936, 1.8933103, 1.8902004, 1.9151701, 1.9089415, 1.9089346, 1.9151921],
                [0.2519119, 0.061573, 0.0494595, 0.0479253, 0.0477197, 0.0479401, 0.048034, 0.0473943, 0.0468735, 0.0468108],
                [0.1915035, 0.0522405, 0.0440092, 0.0438338, 0.0436341, 0.0434896, 0.043353, 0.0433929, 0.0431076, 0.0469716]
            ],
            [
                [1.4169746, 1.4624698, 1.4799298, 1.4960632, 1.4808722, 1.4937343, 1.4921776, 1.4886177, 1.5030005, 1.492997],
                [1.8417474, 1.9136175, 1.968292, 1.9511149, 1.9386106, 1.9417362, 1.9370469, 1.9511076, 1.9495515, 1.9401691],
                [0.2264391, 0.0684185, 0.0541684, 0.0519295, 0.0516059, 0.0510713, 0.0556128, 0.0530082, 0.0519835, 0.0517727],
                [0.1860892, 0.0539773, 0.0441473, 0.0439235, 0.045913, 0.0453405, 0.0453362, 0.0456352, 0.0453773, 0.0441715]
            ],
            [
                [1.799822, 1.8847428, 1.9282357, 1.9231366, 1.9174413, 1.9508356, 1.9413683, 1.9318613, 1.9550918, 1.956064],
                [2.5442903, 2.6396756, 2.7175733, 2.678594, 2.6534723, 2.6803657, 2.6933662, 2.6867981, 2.70779, 2.7169834],
                [0.2387834, 0.086218, 0.0657115, 0.0656308, 0.0653607, 0.0670349, 0.0665059, 0.0666279, 0.0665136, 0.0665525],
                [0.2099016, 0.070507, 0.0568322, 0.0567594, 0.0569891, 0.0568056, 0.0564236, 0.0565652, 0.0561634, 0.056785]
            ]
        ]


class Georgescu(BaseMethod):
    def __init__(self):
        super(Georgescu, self).__init__()
        self.model_name = ["task1", "task2", "task3", "task4"]
        self.tasks = [[0], [1], [2], [3]]
        self.require_sub_model = [[0, 2], [0, 3], [1, 4], [1, 5]]
        self.require_sub_model_all = [[10, 12], [10, 13], [11, 14], [11, 15]]
        self.sub_model_size = [1.591918945, 1.591918945, 0.071289063, 0.071289063, 0.143920898, 0.071289063]
        self.single_task_size = Bytes2Mb(7 * get_image_size(64, 64, 3))
        self.latency = [
            [  # 2
                [0.135990858, ] * 10,
                [0.201506972, ] * 10,
                [0.194757056, 0.045601606, 0.043881512, 0.042294717, 0.041257095, 0.040289712, 0.041384459, 0.04009223, 0.042858768, 0.039830589],
                [0.181886792, 0.044860005, 0.0440382, 0.042821765, 0.041881132, 0.039645624, 0.042106056, 0.041171479, 0.03931706, 0.04142592]
            ],
            [  # 3
                [0.13459971, ] * 10,
                [0.204626083, ] * 10,
                [0.189062619, 0.048365259, 0.043605471, 0.04241128, 0.041698623, 0.041868114, 0.041815138, 0.040962243, 0.041610646, 0.039768434],
                [0.182476449, 0.044616342, 0.043532491, 0.043518043, 0.040133238, 0.040185833, 0.041477394, 0.040636206, 0.039343548, 0.041982532]
            ],
            [  # 4
                [0.152732325, ] * 10,
                [0.201511741, ] * 10,
                [0.204985499, 0.042389655, 0.040116858, 0.038677812, 0.039412808, 0.037670755, 0.038818765, 0.038382888, 0.038452697, 0.03672843],
                [0.195576763, 0.040828538, 0.03947804, 0.039878082, 0.038290191, 0.038246274, 0.038030124, 0.037859988, 0.036315417, 0.037552762]
            ],
            [  # 5
                [0.137135577, ] * 10,
                [0.1890131, ] * 10,
                [0.19721477, 0.040685987, 0.039271116, 0.037487793, 0.036696982, 0.036797166, 0.035172009, 0.035724592, 0.035730457, 0.034275293],
                [0.18819325, 0.041623545, 0.038406134, 0.037783098, 0.036522651, 0.036203599, 0.036601329, 0.03701334, 0.034984875, 0.035346556]
            ]
        ]


def get_model(model_idx) -> BaseMethod:
    models = {
        0: HuangModel(),
        1: ChenModel(),
        2: Georgescu()
    }
    return models.get(model_idx, BaseMethod)


def cal_download_size(model_idx, sub_models):
    model = get_model(model_idx)
    size = model.require_model_size(sub_models)
    return size
