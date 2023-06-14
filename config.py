# coding: UTF-8

import os
import json
import torch
import pandas as pd

#json标签文件解析
def json_jx(js_path):
    #读取映射字典
    with open(js_path, 'r',encoding='utf-8') as fcc_file:
        fcc_data = json.load(fcc_file)
    dicts={}
    for k, v in fcc_data.items():
        dicts.update({vv: k for vv in v})  #字典反转
    return dicts

class Config(object):
    def __init__(self, data_dir):
        assert os.path.exists(data_dir)

        self.train_file = os.path.join(data_dir, "train.csv")
        self.dev_file = os.path.join(data_dir, "dev.csv")
        self.js_path = os.path.join(data_dir, "type.json")

        self.test_file = os.path.join(data_dir, "test.csv")

        # self.label_file = os.path.join(data_dir, "label.txt")
        assert os.path.isfile(self.train_file)
        assert os.path.isfile(self.dev_file)
        # assert os.path.isfile(self.test_file)

        self.saved_model_dir = os.path.join(data_dir, "model")
        self.saved_model = os.path.join(self.saved_model_dir, "bert_model_0602.pth")
        if not os.path.exists(self.saved_model_dir):
            os.mkdir(self.saved_model_dir)

        # self.label_list = [label.strip() for label in open(self.label_file, "r", encoding="UTF-8").readlines()]
        if os.path.isfile(self.js_path):
            self.label_dict = json_jx(self.js_path)
        else:
            self.label_dict = None

        # self.label_list = ['日期','时间','年份','月份','国家','省市','地点','姓名', '性别', '身份证号', '手机号', '座机号/传真','政治面貌','民族', '学历', '专业', 
        #     '公司', '职位', 'uuid', '邮箱','哈希值', '域名','ipv4地址','ipv6地址', 'mac地址', 'url', '用户名','车牌号','信用卡号','银行名称','组织机构代码',
        #     '统一社会信用代码','机关单位','医院','学校','港澳通行证号','台湾通行证号','永久居住证号','中国护照','税务登记证号','医师资格证书编号','医师执业证书编号',
        #     '营业执照','车辆识别代号','公积金号','开户许可证号','银行卡号','军官证号','道路运输经营许可证号','军密认证号',
        #     '药品名称', '药品分类', '药品规格','药品用途', '药品成分', '药品用法', '药品用量', '诊断设备', '医疗器械', '实验室设备', '治疗设备', '护理设备',
        #     '辅助设备', '卫生间设备', '环境设备', '办公设备', '客房设备', '手术类型', '手术名称', '科室名称', '科室介绍',
        #     '医疗设备', '医生职称', '项目名称', '项目描述', '耗材名称', '疾病名称', '手术部位', '手术器械名称']
        self.label_list = list(pd.read_csv(self.dev_file).columns)

        self.num_labels = len(self.label_list)
        self.num_epochs = 2
        self.log_batch = 5
        self.batch_size = 512
        self.max_seq_len = 32
        self.require_improvement = 1000

        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.learning_rate = 5e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# cn = Config(r"D:\pycode\生成代码\my_fake\datas/")
# print(cn.label_list)

