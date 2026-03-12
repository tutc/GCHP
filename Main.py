"""

    
"""
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import tqdm

import numpy as np

import random

import Benchmarks as benchmarks

from typing import Dict, List

from MyConvexHull import ConvexHullND, DataPoint

device = torch.device("cuda")


seed = 317

def set_seed():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def E_distances(X,Y):

    return torch.sqrt( torch.sum(torch.pow(X, 2),dim=1).view(-1,1) -2 * torch.mm(X,Y.T) + torch.sum(torch.pow(Y, 2),dim=1) )


def reduce2D(original):
    if original.ndim == 1:
        mean = original.mean()
        std = original.std(unbiased=False)
        reduced = torch.tensor([mean, std])

    elif original.ndim == 2:
        mean = original.mean(dim=1)               
        std = original.std(dim=1, unbiased=False) 
        reduced = torch.stack([mean, std], dim=1) 
    return reduced

class Main(nn.Module):
    def __init__(self, n_mini_batch, n_class = 10, n_features = 160):
        super(Main, self).__init__()
        self.n_features = n_features        
        self.n_class = n_class              
        self.capacity=3000                
        self.Features = torch.zeros(1, n_features).to(device)   
        self.Labels = torch.zeros(1, self.n_class)              
        self.heso = 0.9

        self.n_mini_batch = n_mini_batch    
        
            
        self.acc_after_each_task=[]         
        self.acc_after_all_task=[]            
        self.forgetting=[]                  

        self.avg_acc_activ=False            

        self.class_to_points: Dict[int, List[DataPoint]] = {}
        self.class_to_hull: Dict[int, ConvexHullND] = {}

        self.reset()                        


    def reset(self):
        self.Features = torch.zeros(1, self.n_features).to(device)
        self.Labels = torch.zeros(1, self.n_class).to(device)        
        self.class_to_points: Dict[int, List[DataPoint]] = {}
        self.class_to_hull: Dict[int, ConvexHullND] = {}

    def forward(self, inputs):
        with torch.no_grad():
            pred = torch.tensor([]).to(device)
            distance = E_distances(inputs,self.Features)        
            soft_norm = F.softmin(distance,dim=-1)       
            pred = torch.matmul(soft_norm, self.Labels)         
        return pred
    
    
    def test_idx(self, test_features, idx_test):
        with torch.no_grad():
            total = 0
            correct = 0
            for idx in idx_test:
                curr_correct = 0
                curr_total = 0
                for batch_idx, (inputs, targets) in enumerate(test_features[idx]):
                    inputs = inputs.to(device)

                    targets = targets.type(torch.LongTensor).to(device)
                    
                    outputs = self.forward(inputs)
                    total += targets.size(0)
                    
                    value_topk, predicted_topk = torch.topk(outputs,1, 1)
                    corr=(predicted_topk.eq(targets.view(-1, 1)).sum().item())
                    
                    curr_correct += corr
                    correct += corr
                    curr_total += targets.size(0)
                
            accuracy = correct/total*100

        return accuracy, curr_correct/curr_total*100


    def storeFeature(self, point, class_id):
        if class_id not in self.class_to_points:
            self.class_to_points[class_id] = [point]
        else:
            if class_id not in self.class_to_hull:
                self.class_to_points[class_id].append(point)
                if len(self.class_to_points[class_id]) >= self.threshold:
                    all_points = self.class_to_points[class_id]
                    self.class_to_hull[class_id] = ConvexHullND(all_points, self.heso)
                    self.class_to_hull[class_id].embedded()
            else:
                self.class_to_hull[class_id].expand(point)
                while len(self.class_to_hull[class_id].origin_hull.vertices) > self.threshold:
                    self.class_to_hull[class_id].shrink()
  
    def updateMemoryBank(self):
        all_features = []
        all_labels = []

        for class_id in self.class_to_points:
            if class_id in self.class_to_hull:
                vert_idx = torch.tensor(self.class_to_hull[class_id].origin_hull.vertices, dtype=torch.long, device=device)
                verts = [self.class_to_hull[class_id].points[i].original for i in vert_idx]
            else:
                verts = [x.original for x in self.class_to_points[class_id]]

            if len(verts) == 0:
                continue 

            # Chuyển list[Tensor] → tensor (n_i, D)
            verts_tensor = torch.stack(verts).to(device)

            n_i = verts_tensor.shape[0]
            one_hot_labels = F.one_hot(
                torch.full((n_i,), class_id, device=device),
                num_classes=self.n_class
            ).float()

            all_features.append(verts_tensor)
            all_labels.append(one_hot_labels)

        if all_features:
            self.Features = torch.cat(all_features, dim=0)
            self.Labels   = torch.cat(all_labels, dim=0)
        else:
            self.Features = torch.empty((0, self.feature_dim), device=device)
            self.Labels   = torch.empty((0, self.n_class), device=device)


    def train_test(self, train_features, test_features):

        acc_test_after_each_task = torch.zeros(len(train_features))
        
        idx_seen = []
        self.avg_acc = []
        with torch.no_grad():
            
            for idx_loader in range(len(train_features)):      
                idx_seen.append(idx_loader)
                for batch_idx, (inputs, targets) in enumerate(train_features[idx_loader]):
                    
                    inputs = inputs.to(device)
                    targets = targets.type(torch.LongTensor).to(device)

                    if batch_idx == self.n_mini_batch:
                        break

                    len_inputs = len(inputs)
                    for idx_x in range(len_inputs):

                        label = int(targets[idx_x])
                        
                        out = inputs.to(device)
                        x = out[idx_x]

                        x_reduced = reduce2D(x)

                        point = DataPoint(x,x_reduced)

                        self.storeFeature(point, label)

                        

                self.updateMemoryBank()

                if self.avg_acc_activ:
                    avg_acc, last_acc = self.test_idx(test_features, idx_seen)

                    self.avg_acc.append(avg_acc)
                    acc_test_after_each_task[idx_loader] = last_acc


            self.acc_after_each_task = acc_test_after_each_task
            acc_test_after_all_task = torch.zeros(len(train_features))
        

            for idx_loader in range(len(test_features)):

                _, last_acc = self.test_idx(test_features, [idx_loader])
                acc_test_after_all_task[idx_loader] = last_acc

                self.acc_after_all_task = acc_test_after_all_task

            return acc_test_after_each_task, acc_test_after_all_task
        
    
    def run_experiment(self, n_mini_batch, train_features, test_features, N_try = 5, random_ordering = True):

        avg_acc = torch.zeros(N_try,len(train_features))
        acc_test_softmin = torch.zeros(N_try,len(train_features))

        self.forgetting = []

        self.n_mini_batch = n_mini_batch

        
        for idx_try in tqdm(range(N_try)):
            self.reset()
            _, acc_last = self.train_test(train_features, test_features)
            # print('KQ last_acc cua la try thu ',idx_try, ' la: ',acc_last)

            if self.avg_acc_activ:
                avg_acc[idx_try] = torch.tensor(self.avg_acc)
                # print('KQ avg_acc cua la try thu ',idx_try, ' la: ',self.avg_acc)
            self.forgetting.append((self.acc_after_each_task*100-self.acc_after_all_task).mean())
            acc_test_softmin[idx_try] = acc_last
            
            
            if random_ordering:
                dataset_shuffle = list(zip(train_features, test_features))
                random.shuffle(dataset_shuffle)
                train_features, test_features = zip(*dataset_shuffle)
        
        # print('KQ Last truoc khi goi mean o trong ham run experiment: ', acc_test_softmin)    
        last_acc = acc_test_softmin.mean(1)

        #print("forgetting softmin inference = {:.1f} % ± {:.1f}".format(np.mean(self.forgetting),np.std(self.forgetting)))
        # print('KQ truoc khi ket thuc ham run experiment')
        # print('Last: ', last_acc)
        # print('Avg: ', avg_acc)
        return avg_acc, last_acc 


def run(data, capacity = 3000, heso = 0.9):

    set_seed()
    
    
    n_mini_batch = 55

    exp = Main(n_mini_batch, n_class = data.n_class, n_features = data.n_features) 
    exp.T = beta 
    exp.capacity = capacity
    exp.heso = heso
    random_ordering = True 
    exp.threshold = (int)(capacity /data.n_class )
    batch_size = min(exp.threshold, 32)
    N_try = 5
    exp.avg_acc_activ = True

    start_time = time.time()
    avg_accuracy, last_accuracy  = exp.run_experiment(n_mini_batch, data.train_features, data.test_features, N_try, random_ordering = random_ordering)
    elapsed_time = round(time.time() - start_time,2)
    avg_acc = round(float(avg_accuracy.mean()),2)
    last_acc = round(float(last_accuracy.mean()),2)    
    mem_size = int(len(exp.Features))

    return avg_acc, last_acc, mem_size, elapsed_time


if __name__ == '__main__':

    dataset = benchmarks.Cifar10Resnet18.CIFAR10RESNET18()
    #dataset = benchmarks.Core50Resnet18.CORE50RESNET18(experiences = 9)
    #dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start = 2, step = 2)
    #dataset = benchmarks.Cifar100Resnet50.CIFAR100RESNET50(start = 5, step = 5)
    #dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start = 2, step = 2)
    #dataset = benchmarks.Cub200Resnet50.CUB200RESNET50(start = 5, step = 5)
    capacity = 3000
    alpha = 0.9
    avg_GCF, last_GCF, meansize_GCF, elapsed_time = run(dataset, capacity = capacity, heso = alpha)
    print(avg_GCF)
    print(last_GCF)
    print(meansize_GCF)
    print(elapsed_time)



    
