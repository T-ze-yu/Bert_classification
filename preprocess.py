# coding: UTF-8

import time
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class DataProcessor(object):
    def __init__(self, cnf, tokenizer, seed, mode):
    # def __init__(self, path, label_list, label_dict, device, tokenizer, batch_size, max_seq_len, seed):
        self.seed = seed
        self.mode = mode
        self.device = cnf.device
        self.tokenizer = tokenizer
        self.batch_size = cnf.batch_size
        self.max_seq_len = cnf.max_seq_len
        
        self.label_dict, self.label_list = cnf.label_dict, cnf.label_list
        if mode=='train':
            self.data , self.label_count= self.load_tt(cnf.train_file)
        elif mode=='dev':
            self.data , self.label_count= self.load_tt(cnf.dev_file)
        elif mode=='test':
            self.data= self.load_test(cnf.test_file)
            self.columns = self.data.columns
        else:
            raise('mode参数错误')
        # self.data = self.load(path)

        self.index = 0
        self.residue = False
        if mode=='test':
            self.num_samples = len(self.data)
        else:
            self.num_samples = len(self.data[0])
        self.num_batches = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.residue = True

    def load(self, path):
        contents = []
        labels = []
        with open(path, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:    continue
                if line.find('\t') == -1:   continue
                content, label = line.split("\t")
                contents.append(content)
                labels.append(int(label))
        #random shuffle
        index = list(range(len(labels)))
        random.seed(self.seed)
        random.shuffle(index)
        contents = [contents[_] for _ in index]
        labels = [labels[_] for _ in index]
        return (contents, labels)

    def load_test(self, path):
        df = pd.read_csv(path)
        # df = pd.read_csv(path).sample(n=100)
        for cc in df.columns:
            # print(df[cc].values.dtype)
            # if df[cc].values.dtype == 'int64' or df[cc].values.dtype == 'uint64': 
            #     df.drop([cc],axis=1,inplace=True)
            if df[cc].values.dtype != 'object':
                df[cc] = [str(i) for i in df[cc].values]
        return df

    #加载csv文件输出一个label为key,值为数据列表的数据字典
    def load_csv(self, path):
        datas_dict = {}
        df = pd.read_csv(path)
        i=-1
        for cc in df.columns:      #遍历表字段
            i+=1
            if cc in self.label_dict.keys():     #如果在标签字典则进行映射
                lab = self.label_dict[cc]
                data = []
                for cont in df.values[1:,i]:     #遍历该列文本值
                    if cont!='':        #文本值不为空
                        data.append(cont)
                if lab in datas_dict.keys():
                    datas_dict[lab].extend(data)
                else:
                    datas_dict.update({lab:data})
        return datas_dict

    #输入数据字典和抽取数量，返回混淆的数据和标签
    # def confusion_datas(self, datas_dict, Sample_size):
    #     contents = []
    #     labels = []

    def load_tt(self, path):
        contents = []
        labels = []
        label_count = {}        #标签数量统计
        df = pd.read_csv(path)
        columns = df.columns
        
        i=-1
        for cc in columns:      #遍历表字段
            i+=1
            if self.label_dict and cc in self.label_dict.keys():     #如果在标签字典则进行映射
                lab = self.label_dict[cc]
            elif not self.label_dict:
                lab = cc
            else:
                continue

            sy = self.label_list.index(lab)
            sum = 0
            # print(df.values[:,i])
            for cont in df.values[:,i]:     #遍历该列文本值
                if cont!='':        #文本值不为空
                    contents.append(str(cont))
                    labels.append(sy)
                    sum+=1
            if lab in label_count.keys():
                label_count[lab] += sum
            else:
                label_count.update({lab:sum})

        #random shuffle
        # print(contents)
        index = list(range(len(labels)))
        random.seed(self.seed)
        random.shuffle(index)
        contents = [contents[_] for _ in index]
        labels = [labels[_] for _ in index]
        # print(label_count)
        return (contents, labels), label_count

    def __next__(self):
        if self.mode=='test':
            if self.index < len(self.columns):
                batch_y = self.columns[self.index]
                batch_x =  self.data[batch_y].dropna().values
                self.index += 1
                batch = self._to_tensor(batch_x, batch_y)
                return batch
            self.index = 0
            raise StopIteration()

        elif self.residue and self.index == self.num_batches:
            batch_x = self.data[0][self.index * self.batch_size: self.num_samples]
            batch_y = self.data[1][self.index * self.batch_size: self.num_samples]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch
        elif self.index >= self.num_batches:
            self.index = 0
            raise StopIteration
        else:
            batch_x = self.data[0][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch_y = self.data[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch

    def _to_tensor(self, batch_x, batch_y):
        inputs = self.tokenizer.batch_encode_plus(
            batch_x,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation="longest_first",
            return_tensors="pt")
        inputs = inputs.to(self.device)
        if isinstance(batch_y,str):
            labels = batch_y
        else:
            labels = torch.LongTensor(batch_y).to(self.device)
        return (inputs, labels)

    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        elif self.mode=='test':
            return len(self.columns)
        else:
            return self.num_batches

if __name__=='__main__':
    def json_jx(js_path):
        #读取映射字典
        with open(js_path, 'r',encoding='utf-8') as fcc_file:
            fcc_data = json.load(fcc_file)
        dicts={}
        for k, v in fcc_data.items():
            dicts.update({vv: k for vv in v})  #字典反转
        label_list = list(fcc_data.keys())
        return dicts, label_list

    def load_test(path, label_dict, label_list):
        contents = []
        labels = []
        label_count = {}        #标签数量统计
        df = pd.read_excel(path)
        columns = df.columns
        
        i=-1
        for cc in columns:      #遍历表字段
            i+=1
            if cc in label_dict.keys():     #如果在标签字典则进行映射
                lab = label_dict[cc]
                sy = label_list.index(lab)
                sum = 0
                for cont in df.values[1:,i]:     #遍历该列文本值
                    if cont!=' ':        #文本值不为空
                        contents.append(str(cont))
                        labels.append(sy)
                        sum+=1
                if lab in label_count.keys():
                    label_count[lab] += sum
                else:
                    label_count.update({lab:sum})
        return contents, labels, label_count

if __name__=='__main__':
    dicts, label_list = json_jx(r'E:\datas\分类分级\test1/type.json')
    contents, labels, label_count = load_test(r'E:\datas\分类分级\test1/all_senstype.xlsx', dicts, label_list)
    print(contents[:5])
    print(labels[:5])
    print(label_count)