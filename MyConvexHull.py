import torch

from typing import List
from scipy.spatial import ConvexHull


class DataPoint:

    def __init__(self, original: torch.Tensor, reduced=None):
        self.original = original
        if reduced is not None:
            self.reduced = reduced.to(original.device)
        else:
            self.reduced = None

    def embedded(self, y, alpha = 0.5):
        self.original = self.original*alpha + y.original*(1-alpha)
        #print('Alpha tinh embedded = ',alpha)
    @staticmethod
    def midpoint(x, y):
        original = x.original*0.5 + y.original*0.5
        reduced = x.reduced*0.5 + y.reduced*0.5
        return DataPoint(original,reduced)
            
    

class ConvexHullND:

    def __init__(self, points: List[DataPoint], alpha = 0.5):

        self.points = points
        self.device = points[0].original.device
    
        #self.hull_points, self.inner = self._build_hull(self.points)
        self.origin_hull, self.inner_idx = self._build_hull(self.points)
        self.alpha = alpha
        #print('Alpha cua ConvexHull = ',alpha)
    def set_reduced_from_model(self, model):
        for p in self.points:
            p.reduced = model.transform(p.original)

    def process_and_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # Huấn luyện PCA online
        self.pca.partial_fit(x)
        # Giảm chiều nếu đã đủ batch
        return self.pca.transform(x)
    
    def _build_hull(self, points: List[DataPoint]):
    
        reduced_array = torch.stack([p.reduced.squeeze(0).to(self.device) for p in points])  # (N, 2)
        hull = ConvexHull(reduced_array.cpu().numpy(), incremental = True)
        vert_idx = torch.tensor(hull.vertices, dtype=torch.long, device=self.device)
        all_idx = torch.arange(reduced_array.shape[0], device=self.device)
        mask = torch.ones(reduced_array.shape[0], dtype=bool, device=self.device)
        mask[vert_idx] = False
        inner_idx = all_idx[mask]
        return hull, inner_idx


    def embedded(self):
        #print('Len of inner = ', len(self.inner_idx))
        inner =  [self.points[i] for i in self.inner_idx]
        hull_points = [self.points[i] for i in self.origin_hull.vertices]  
        i = 1        
        #change = []        
        for x in inner:
            #print('inner thu = ', i)
            i = i + 1

            #hull_tensor = torch.stack([p.reduced.to(self.device) for p in hull_points])  # (H, dim)   Version 2 draft manuscript uses this metric
            #dists = torch.norm(hull_tensor - x.reduced.unsqueeze(0).to(self.device), dim=1)  # (H,)


            hull_tensor = torch.stack([p.original.to(self.device) for p in hull_points])  # (H, dim)
            dists = torch.norm(hull_tensor - x.original.unsqueeze(0).to(self.device), dim=1)  # (H,)

            nearest_idx = torch.argmin(dists).item()
            #change.append(self.origin_hull.vertices[nearest_idx])
            #print('Truoc embedded ',self.points[self.origin_hull.vertices[nearest_idx]].original)
            self.points[self.origin_hull.vertices[nearest_idx]].embedded(x,self.alpha)
            #print('Sau embedded   ',self.points[self.origin_hull.vertices[nearest_idx]].original)
            #input('Embedded xong 1 phan tu')
        self.inner_idx = []  # reset
        #return change


    def shrink(self):     #Reduce dựa vào điểm xa nhất trên không gian 2D

        hull_points = [self.points[i] for i in self.origin_hull.vertices]
        hull_reduced = torch.stack([p.reduced.to(self.device) for p in hull_points])
        centroid = hull_reduced.mean(dim=0)

        dists = torch.norm(hull_reduced - centroid, dim=1)

        far_idx = torch.argmax(dists)
        x = hull_points[far_idx]

        remaining_pts = hull_points[:far_idx] + hull_points[far_idx+1:]

        reduced_remaining = torch.stack([p.reduced.to(self.device) for p in remaining_pts])  # shape: (N, 2)
        dists_to_x = torch.norm(reduced_remaining - x.reduced.to(self.device), dim=1)

        nearest_idx = torch.argmin(dists_to_x)
        y = hull_points[nearest_idx]

        x_mean = DataPoint.midpoint(x,y)
        
        self.points = hull_points   #Mới thêm vào cho chuẩn

        self.points[far_idx] = x_mean  # cập nhật vị trí điểm mới
        self.points.pop(nearest_idx)

        # self.points[self.origin_hull.vertices[far_idx]] = x_mean  # cập nhật vị trí điểm mới  #Nếu không đổ hull_points lại cho self.points thì mới dùng 2 dòng này
        # self.points.pop(self.origin_hull.vertices[nearest_idx])


        reduced_array = torch.stack([p.reduced.squeeze(0).to(self.device) for p in self.points])  # (N, 2)
        self.origin_hull = ConvexHull(reduced_array.cpu().numpy(), incremental = True)

    def shrink_original(self):        #Reduce dựa vào điểm xa nhất trên không gian gốc

        hull_points = [self.points[i] for i in self.origin_hull.vertices]
        #hull_reduced = torch.stack([p.reduced.to(self.device) for p in hull_points])
        
        hull_original = torch.stack([p.original.to(self.device) for p in hull_points])
        
        #centroid = hull_reduced.mean(dim=0)
        centroid = hull_original.mean(dim=0)

        #dists = torch.norm(hull_reduced - centroid, dim=1)
        dists = torch.norm(hull_original - centroid, dim=1)

        far_idx = torch.argmax(dists)
        x = hull_points[far_idx]

        #remaining_pts = hull_points[:far_idx] + hull_points[far_idx+1:]

        #reduced_remaining = torch.stack([p.reduced.to(self.device) for p in remaining_pts])  # shape: (N, 2)
        #original_remaining = torch.stack([p.original.to(self.device) for p in remaining_pts])  # shape: (N, 2)

        
        idx_left = (far_idx - 1) % len(hull_points)
        idx_right = (far_idx + 1) % len(hull_points)
        y_left = hull_points[idx_left]
        y_right = hull_points[idx_right]

        dx_left = torch.norm(
            x.original.to(self.device) - y_left.original.to(self.device)
        )

        dx_right = torch.norm(
            x.original.to(self.device) - y_right.original.to(self.device)
        )

        if dx_left < dx_right:
            y = y_left
            idx_y = idx_left
        else:
            y = y_right
            idx_y = idx_right

        #dists_to_x = torch.norm(reduced_remaining - x.reduced.to(self.device), dim=1)       #Tìm gần nhất trên 2D để đảm bảo đó là đỉnh kề, nếu không thì sẽ phá vỡ bao lồi
        #dists_to_x = torch.norm(original_remaining - x.original.to(self.device), dim=1)

        #nearest_idx = torch.argmin(dists_to_x)
        #y = hull_points[nearest_idx]

        x_mean = DataPoint.midpoint(x,y)
        #print('Len self.points truoc khi gan lai tu hull_points =',len(self.points))
        

        self.points = hull_points
        #print('Len self.points sau khi gan lai tu hull_points (truoc khi pop) =',len(self.points))

        #self.points[self.origin_hull.vertices[far_idx]] = x_mean  # cập nhật vị trí điểm mới
        #self.points.pop(self.origin_hull.vertices[nearest_idx])
        

        self.points[far_idx] = x_mean  # cập nhật vị trí điểm mới
        #self.points.pop(nearest_idx)
        self.points.pop(idx_y)

        #print('Len self.points sau pop =',len(self.points))
        
        #reduced_array = torch.stack([p.reduced.squeeze(0).to(self.device) for p in self.points])  # (N, 2)
        
        reduced_array = torch.stack([p.reduced.squeeze(0).to(self.device) for p in self.points])  # (N, 2)
        
        #print('Len hull truoc tao lai =',len(self.origin_hull.vertices))
        self.origin_hull = ConvexHull(reduced_array.cpu().numpy(), incremental = True)
        #print('Len cua reduced_array khi tao lai hull = ',len(reduced_array.cpu().numpy()))
        #print('Len hull sau tao lai =',len(self.origin_hull.vertices))
        #print('=============')
    
    def expand(self, new_point: DataPoint):
        """
        Thêm điểm mới vào bao lồi (không gọi lại ConvexHull)
        """
        self.points.append(new_point)
        self.origin_hull.add_points(new_point.reduced.unsqueeze(0).cpu().numpy())
        
        vert_idx = torch.tensor(self.origin_hull.vertices, dtype=torch.long, device=self.device)
        all_idx = torch.arange(len(self.points), device=self.device)
        mask = torch.ones(len(self.points), dtype=bool, device=self.device)
        mask[vert_idx] = False
        self.inner_idx = all_idx[mask]
        self.embedded()





if __name__ == '__main__':
    # Khởi tạo bao lồi trong không gian 3 chiều
    points = torch.rand((100, 512), device='cuda')
    hull = ConvexHullND(points)

    # Mở rộng với điểm mới
    new_point = torch.rand(512, device='cuda')

    print('Expand')
    hull.expand(new_point)

    # Kéo đỉnh
    #hull.pull_vertex_Old() #Ok rồi nhưng hơi chậm do gọi lại ConvexHull
    print('Pull')
    hull.pull_vertex()
    print('Shrink')
    # Thu nhỏ
    hull.shrink()
    print('Done')
