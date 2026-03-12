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

    @staticmethod
    def midpoint(x, y):
        original = x.original*0.5 + y.original*0.5
        reduced = x.reduced*0.5 + y.reduced*0.5
        return DataPoint(original,reduced)
            
    

class ConvexHullND:

    def __init__(self, points: List[DataPoint], alpha = 0.5):

        self.points = points
        self.device = points[0].original.device
    
        self.origin_hull, self.inner_idx = self._build_hull(self.points)
        self.alpha = alpha

    def set_reduced_from_model(self, model):
        for p in self.points:
            p.reduced = model.transform(p.original)

    
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
        inner =  [self.points[i] for i in self.inner_idx]
        hull_points = [self.points[i] for i in self.origin_hull.vertices]  
        i = 1        
       
        for x in inner:

            i = i + 1

            hull_tensor = torch.stack([p.original.to(self.device) for p in hull_points])  # (H, dim)
            dists = torch.norm(hull_tensor - x.original.unsqueeze(0).to(self.device), dim=1)  # (H,)

            nearest_idx = torch.argmin(dists).item()

            self.points[self.origin_hull.vertices[nearest_idx]].embedded(x,self.alpha)

        self.inner_idx = []  # reset
        #return change

    def shrink(self):       

        hull_points = [self.points[i] for i in self.origin_hull.vertices]
        
        hull_original = torch.stack([p.original.to(self.device) for p in hull_points])
        
        centroid = hull_original.mean(dim=0)

        dists = torch.norm(hull_original - centroid, dim=1)

        far_idx = torch.argmax(dists)
        x = hull_points[far_idx]

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

        x_mean = DataPoint.midpoint(x,y)
  
        self.points = hull_points
 
        self.points[far_idx] = x_mean  # cập nhật vị trí điểm mới
        
        self.points.pop(idx_y)

        
        reduced_array = torch.stack([p.reduced.squeeze(0).to(self.device) for p in self.points])  # (N, 2)
        
        self.origin_hull = ConvexHull(reduced_array.cpu().numpy(), incremental = True)
 
    def expand(self, new_point: DataPoint):
        self.points.append(new_point)
        self.origin_hull.add_points(new_point.reduced.unsqueeze(0).cpu().numpy())
        
        vert_idx = torch.tensor(self.origin_hull.vertices, dtype=torch.long, device=self.device)
        all_idx = torch.arange(len(self.points), device=self.device)
        mask = torch.ones(len(self.points), dtype=bool, device=self.device)
        mask[vert_idx] = False
        self.inner_idx = all_idx[mask]
        self.embedded()





if __name__ == '__main__':
    points = torch.rand((100, 512), device='cuda')
    hull = ConvexHullND(points)

    new_point = torch.rand(512, device='cuda')

    hull.expand(new_point)

    hull.pull_vertex()
    hull.shrink()
    print('Done')
