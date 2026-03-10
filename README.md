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
    python General_main.py --dataset cifar10 --backbone resnet18 --capacity 1000
  </code>
</pre>
##### Dataset: CORe-50, Feature extractor: Resnet-18, Memory size: 2000
<pre>
  <code id="code-snippet">
    python General_main.py --dataset core50 --backbone resnet18 --memory 2000
  </code>
</pre>
##### Dataset: Split CIFAR-100, Feature extractor: Resnet-50, Step: 2
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cifar100 --backbone resnet50 --step 2
  </code>
</pre>
##### Dataset: Split CUB-200, Feature extractor: Resnet-50, Step: 5
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cub200 --backbone resnet50 --step 5
  </code>
</pre>
