# Geometry-Controlled Convex Hull Prototype Framework for Online Task-Free Continual Learning
## Dataset
- Split CIFAR-10
- Split CIFAR-100
- Split CUB-200
- CORe-50
## Feature extractor
- Resnet-18
- Resnet-50
## Sample commands to run GCHP
##### Dataset: Split CIFAR-10, Feature extractor: Resnet-18, Prototype capacity: 1000
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cifar10 --backbone resnet18 --capacity 1000
  </code>
</pre>
##### Dataset: CORe-50, Feature extractor: Resnet-18, Prototype capacity: 2000
<pre>
  <code id="code-snippet">
    python General_main.py --dataset core50 --backbone resnet18 --capacity 2000
  </code>
</pre>
##### Dataset: Split CIFAR-100, Step: 2, Feature extractor: Resnet-50, Prototype capacity: 3000 
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cifar100 --step 2 --backbone resnet50
  </code>
</pre>
##### Dataset: Split CUB-200, Step: 5, Feature extractor: Resnet-50, Prototype capacity: 5000 
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cub200 --step 5 --backbone resnet50 --capacity 5000
  </code>
</pre>
