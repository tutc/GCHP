"""
    Dùng bao lồi có embedded các node bên trong nhưng chỉ embedded dữ liệu gốc, Khi reduced thì tìm đỉnh xa nhất dựa trên không gian gốc chứ không phải không gian 2D, bộ nhớ cân bằng, ngăn overload dữ liệu bằng scale bao lồi, inference bằng Euclidean
    Mỗi khi có điểm mới vô thì incremental expand, sau đó embedded lại 1 lần (nếu cần), nếu vượt ngưỡng thì shrink.
        
    Nhận thấy Dim = 7 là không khả thi vì quá lâu, Dim = 6 thì acc có cao hơn là không nhiều so với thời gian tính toán. Dim = 2 thì không tốt lắm.
    Dim = 3, 4 thì đẹp về tốc độ. Dim = 5 thì acc cao hơn 3, 4 và có thể chấp nhận được về thời gian tính toán.
    Kết luận: Nên báo cáo Dim = 4. Nếu acc của Dim = 4 không đủ so sánh thì dùng Dim = 5.   
    
    
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
        # Nếu là 1D: ta tính luôn mean và std scalar
        mean = original.mean()
        std = original.std(unbiased=False)
        #reduced = torch.from_numpy(torch.tensor([mean, std]))  # shape: (2,)
        reduced = torch.tensor([mean, std])
        #reduced = vec.numpy()

    elif original.ndim == 2:
        # Nếu là 2D: Tính mean, std theo chiều 1
        mean = original.mean(dim=1)               # shape: (N,)
        std = original.std(dim=1, unbiased=False) # shape: (N,)
        reduced = torch.stack([mean, std], dim=1)  # shape: (N, 2)
        #reduced = torch.from_numpy(torch.stack([mean, std]), dim=1).squeeze(0)  # shape: (N, 2)
    #torch.from_numpy(reduced).squeeze(0)
    return reduced

class Main(nn.Module):
    def __init__(self, n_mini_batch, n_class = 10, n_features = 160):
        super(Main, self).__init__()
        self.n_features = n_features        #Số chiều (kích thước) của 1 features
        self.n_class = n_class              #Số lớp cần phân loại
        self.memorySize=3000                #Kích thước bộ nhớ cho phép lưu trữ dữ liệu để học
        self.Features = torch.zeros(1, n_features).to(device)   #Lưu thông tin dữ liệu sau khi trích đặc trưng
        self.Labels = torch.zeros(1, self.n_class)              #Nhãn của dữ liệu tương ứng
        self.heso = 0.9

        self.n_mini_batch = n_mini_batch    #Số batch tối đa được dùng để train, nhiều quá thì chậm mà còn không hiệu quả, ít quá thì không đủ học.
        
            
        self.acc_after_each_task=[]         #Để tính forgetting
        self.acc_after_all_task=[]            #Để tính forgetting
        self.forgetting=[]                  #Cần thì in ra, hiện tại thì không dùng

        self.avg_acc_activ=False            #Để biết sau khi train xong 1 task thì có test liền không

        self.class_to_points: Dict[int, List[DataPoint]] = {}
        self.class_to_hull: Dict[int, ConvexHullND] = {}

        self.reset()                        #Mỗi thí nghiệm thì chạy N_try lần, mỗi lần chạy thì reset bộ nhớ lưu trữ dữ liệu


    def reset(self):
        self.Features = torch.zeros(1, self.n_features).to(device)
        self.Labels = torch.zeros(1, self.n_class).to(device)        
        self.class_to_points: Dict[int, List[DataPoint]] = {}
        #self.class_to_points: Dict[int, List[torch.Tensor]] = {}
        self.class_to_hull: Dict[int, ConvexHullND] = {}

    def forward(self, inputs):
        
        with torch.no_grad():
            pred = torch.tensor([]).to(device)
            distance = E_distances(inputs,self.Features)        #Tính khoảng cách từ inputs đến tất cả features được lưu trữ trong bộ nhớ
            soft_norm = F.softmin(distance,dim=-1)       #Chuẩn hóa, khoảng cách càng nhỏ thì cho ra giá trị càng lớn
            pred = torch.matmul(soft_norm, self.Labels)         #Nhân ma trận, tính khả năng 1 input thuộc về label nào
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

                    #inputs = reduce2D(inputs).to(device)

                    targets = targets.type(torch.LongTensor).to(device)
                    #print(targets)
                    outputs = self.forward(inputs)
                    total += targets.size(0)
                    
                    value_topk, predicted_topk = torch.topk(outputs,1, 1)
                    corr=(predicted_topk.eq(targets.view(-1, 1)).sum().item())
                    
                    curr_correct += corr
                    correct += corr
                    curr_total += targets.size(0)
                # print('Acc cua task ',idx, ' la:', curr_correct/curr_total*100)
            accuracy = correct/total*100

        return accuracy, curr_correct/curr_total*100


    def storeFeature(self, point, class_id):
        #print('He so = ',self.heso)
        if class_id not in self.class_to_points:
            # Nếu chưa có lớp → dùng điểm đầu tiên để khởi tạo hull
            self.class_to_points[class_id] = [point]
            #print('to points cho diem dau tien cua lop ', class_id)
        else:
            if class_id not in self.class_to_hull:
                self.class_to_points[class_id].append(point)
                if len(self.class_to_points[class_id]) >= self.threshold:
                    #print('Lop ',class_id, ' da co ',len(self.class_to_points[class_id]), 'phan tu, da vuot nguong ', self.threshold, '. Tim bao loi lan dau')
                    #all_points = torch.stack(self.class_to_points[class_id])
                    all_points = self.class_to_points[class_id]
                    #print('Len all points truoc khi tim bao loi = ', len(all_points))
                    self.class_to_hull[class_id] = ConvexHullND(all_points, self.heso)
                    self.class_to_hull[class_id].embedded()
                    #print('Len bao loi = ', len(self.class_to_hull[class_id].origin_hull.vertices))
                    #change = self.class_to_hull[class_id].embedded()
                    #print('Cac dinh bi thay doi')
                    #for c in change:
                    #    print('Sau embedded   ', self.class_to_hull[class_id].points[c].original)
                    #    input()           
            else:
                #Expand
                #all_points = self.class_to_hull[class_id].points
                #all_points.append(point)             
       
                #self.class_to_hull[class_id] = ConvexHullND(all_points)  
                #self.class_to_hull[class_id].embedded()
     		
                #print('Len cua class to hull sau khi expand cho lop ',class_id, ' la ', len(self.class_to_hull[class_id].vert))
                #input()
                #print('Len bao loi lop ',class_id, ' truoc expand = ', len(self.class_to_hull[class_id].origin_hull.vertices))
                self.class_to_hull[class_id].expand(point)
                #print('Len bao loi lop ',class_id, ' sau expand = ', len(self.class_to_hull[class_id].origin_hull.vertices))
                i=1
                while len(self.class_to_hull[class_id].origin_hull.vertices) > self.threshold:
                    #print('Len bao loi lop ',class_id, ' truoc shrink = ', len(self.class_to_hull[class_id].origin_hull.vertices))
                    #print('Nguong la ', self.threshold)
                    #print('Len feature trước khi shrink = ',int(len(self.Features)))
                    
                    #self.class_to_hull[class_id].shrink()
                    #print('Trước khi thuc hien contraction lan thu',i ,'với len =',len(self.class_to_hull[class_id].origin_hull.vertices) )
                    self.class_to_hull[class_id].shrink_original()
                    #print('Sau khi thuc hien contraction lan thu',i ,'với len =',len(self.class_to_hull[class_id].origin_hull.vertices) )
                    #print('------------')
                    i+=1
                    #print('Len feature sau khi shrink = ',int(len(self.Features)))
                    #print('Len bao loi lop ',class_id, ' sau shrink = ', len(self.class_to_hull[class_id].origin_hull.vertices))
                    
            #
            # input('Vưa store xong 1 diem. Nhan 1 phim de tiep tuc')    

    def updateMemoryBank(self):
        all_features = []
        all_labels = []

        for class_id in self.class_to_points:
            if class_id in self.class_to_hull:
                vert_idx = torch.tensor(self.class_to_hull[class_id].origin_hull.vertices, dtype=torch.long, device=device)
                verts = [self.class_to_hull[class_id].points[i].original for i in vert_idx]
                #verts = [x.original for x in self.class_to_hull[class_id].points[].hull]
            else:
                verts = [x.original for x in self.class_to_points[class_id]]

            if len(verts) == 0:
                continue  # bỏ qua nếu không có điểm nào

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

        #print('Len features = ',int(len(self.Features)))

    def train_test(self, train_features, test_features):

        acc_test_after_each_task = torch.zeros(len(train_features))
        
        idx_seen = []
        self.avg_acc = []
        with torch.no_grad():
            
            for idx_loader in range(len(train_features)):       #Mỗi idx_loader là một task
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
                        #point = DataPoint(x_reduced,x_reduced)

                        self.storeFeature(point, label)

                        # self.Features = torch.cat((self.Features,x.view(1,-1)))
                        # targets_one_hot = F.one_hot(targets[idx_x], num_classes=self.n_class).float()
                        # self.Labels = torch.cat((self.Labels,targets_one_hot.view(1,-1)))
                        
                        # self.get_ConvexHull_Balance(label)
                        

                self.updateMemoryBank()

                if self.avg_acc_activ:
                    avg_acc, last_acc = self.test_idx(test_features, idx_seen)
                    # print('Idx seen la: ', idx_seen)
                    # print('Avg acc cua task thu ',idx_loader, ' la: ', avg_acc)
                    # print('Last acc cua task thu ',idx_loader, ' la: ', last_acc)

                    self.avg_acc.append(avg_acc)
                    acc_test_after_each_task[idx_loader] = last_acc


            self.acc_after_each_task = acc_test_after_each_task
            # print('Acc after each task la (lay tu last cua tung task): ', acc_test_after_each_task)

            acc_test_after_all_task = torch.zeros(len(train_features))
        
            # test after learn all task

            for idx_loader in range(len(test_features)):

                _, last_acc = self.test_idx(test_features, [idx_loader])
                acc_test_after_all_task[idx_loader] = last_acc
                # print('Last acc sau khi test tung ([idx_loader]): ',[idx_loader], 'la: ', last_acc)
                self.acc_after_all_task = acc_test_after_all_task

            # print('Acc after learn all task la (lay tu last cua tung task): ', acc_test_after_all_task)
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


def run(data, memorysize = 3000, heso = 0.9):

    set_seed()
    
    
    n_mini_batch = 55

    exp = Main(n_mini_batch, n_class = data.n_class, n_features = data.n_features) 
    exp.T = beta 
    exp.memorySize = memorysize
    exp.heso = heso
    random_ordering = True 
    exp.threshold = (int)(memorysize /data.n_class )
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
    memorySize = 3000
    alpha = 0.9
	#avg_GCF, last_GCF, meansize_GCF, elapsed_time = run(dataset, beta=1, memorysize = memorySize, dim=dim, heso = alpha, reduce = '2D')
    avg_GCF, last_GCF, meansize_GCF, elapsed_time = run(dataset, memorysize = memorySize, heso = alpha)
    print(avg_GCF)
    print(last_GCF)
    print(meansize_GCF)
    print(elapsed_time)



    
