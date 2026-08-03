# Geometry-Controlled Convex Hull Prototype Framework for Online Task-Free Continual Learning
## Abstract
Online task-free continual learning requires models to learn from a non-stationary data stream without task boundaries, replay buffers, or multiple passes, while preserving previously acquired knowledge. Existing exemplar-free methods either rely on unstable geometric assumptions or synthetic feature generation, which may degrade under highly non-stationary streams. To address these limitations, we propose GCHP, a Geometry-Controlled Convex Hull Prototype framework for online task-free continual learning.
GCHP represents each class by maintaining a convex hull in a low-dimensional control space, together with its corresponding semantic prototypes in the original feature space. Upon receiving new samples, the hull is updated via expansion and contraction in the control space, and regulated prototypes are embedded into the original feature space for inference. This geometry-controlled mechanism enables stable boundary refinement under bounded prototype capacity without storing real exemplars.
Extensive experiments on CIFAR-10, CIFAR-100, CORe-50, and CUB-200 show that GCHP outperforms prior approaches, despite operating in a strictly single-pass setting without using memory buffers. Ablation studies further demonstrate the importance of the geometric representation and the robustness of the prototype capacity.
These results underline the effectiveness of geometric consolidation for continual learning and highlight GCHP as a simple, stable, and scalable alternative for online exemplar-free scenarios.
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
## Citation
If you use this code in your research, please cite the following relevant work:
<pre>
  <code id="code-snippet">
    @article{tran2026geometry,
      title={Geometry-controlled convex hull prototype framework for online task-free continual learning},
      author={Tran, Cong Tu and Nguyen, Thanh Tuan and Nguyen, Thanh Phuong and Thirion-Moreau, Nad{\`e}ge},
      journal={Neurocomputing},
      pages={134619},
      year={2026},
      publisher={Elsevier}
    }  </code>
</pre>
