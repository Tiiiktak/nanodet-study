# NanoDet 代码解读

NanoDet is a FCOS-style one-stage anchor-free object detection model which using ATSS for target sampling and using Generalized Focal Loss for classification and box regression.

![img](https://github.com/RangiLyu/nanodet/raw/main/docs/imgs/Model_arch.png)

> - [YOLO之外的另一选择，手机端97FPS的Anchor-Free目标检测模型NanoDet现已开源~](https://zhuanlan.zhihu.com/p/306530300)
> - [FCOS:一阶全卷积目标检测](https://zhuanlan.zhihu.com/p/63868458)
> - [大白话 Generalized Focal Loss](https://zhuanlan.zhihu.com/p/147691786)
> - [深度学习_YOLO与SSD(6)](https://blog.csdn.net/qq_31784189/article/details/112723635)
> - [YOLO详解](https://zhuanlan.zhihu.com/p/25236464/)
> - [bounding box回归的原理学习——yoloV1](https://blog.csdn.net/brightming/article/details/78072045)

## Train

```
1.trainer.py:
	run_epoch() -> run_step(model, meta, mode='train') -> model.module.forward_train(meta)
2.one_stage.py
	self(gt_meta['img']) -> forward(gt_meta['img']) 
	-> self.backbone(x) -> self.fpn(x) -> self.head(x)
3.nanodet_head.py
	self.forward(feats) -> self.forward_single() -> return to one_stage.forward_train()
4.one_stage.py
	self.head.loss(preds, gt_meta)
5.gfl_head.py
	loss(preds, gt_meta)
```



## GFocal Loss

> https://zhuanlan.zhihu.com/p/147691786
>
> https://arxiv.org/pdf/2006.04388.pdf

![image-20210315143105506](C:%5CUsers%5C1%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210315143105506.png)

- 为保持traning和test的一致，将分类和质量预测的score合并，同时兼顾正负样本
  - 保留分类的向量，但是对应类别位置的置信度的物理含义不再是分类的score，而是改为质量预测的score

### QFL

$$
QFL(\sigma)=-|y-\sigma|^\beta ((1-y)\log(1-\sigma)+y\log(\sigma)) \\
质量标签\ y \in [0, 1] \\
预测\ \sigma \\
\beta =2 最优
$$

### DFL

以类似交叉熵的形式去优化与标签y最接近的一左一右两个位置的概率，从而让网络快速地聚焦到目标位置的邻近区域的分布中去。
$$
DFL(S_i, S_{i+1})=-((y_{i+1}-y)\log(S_i)+(y-y_i)\log(S_{i+1})) \\
$$


### Integral

传入`4 * (reg_max + 1)`分量，与`4 * [0, 1, ... 7]`积分，获得四个方向上`l, r, t, b`

## ATSS

using ATSS for target sampling  

使用ATSS将`ground truth`分配到对应的`bboxes(grid cells)`上，进行后续的回归

`atss_assigner.py -> assign()`

分配流程：

1. 计算各个`gt`与`bboxes(各level)`的`iou`
2. 计算各个`gt`与`bboxes(各level)`的中心点距离
3. 在每个`level`上，为每个`gt`选取`k`个与其中心距离最短的`bbox`，作为`candidates`
4. 得到每个`gt`的`candidates`的`iou`，将`mean + std`作为`iou`阈值
5. 使用`mean + std`阈值筛选`candidates`
6. 将正样本的中心限制在`gt`中
   - 仅保留`candidates`中心位于`gt`中的以及框内部分
   - 其余均作为负样本
   - 若一个`bbox`被分配了多个`gt`，保留`iou`最大的

理解：

- 在不同尺度特征中，分别选取不同size的grid，且grid之间间隔为cell_size的一半，得以覆盖整张feat
- 分配与target距离最近、iou最小的grid

> https://blog.csdn.net/sinat_37532065/article/details/105126367

## grid_cells

`get_grid_cells()`中，若`featmap_size=(2, 3), scale=5, stride=8`，则

```matlab
y = tensor([ 4.,  4.,  4., 12., 12., 12.])
x = tensor([ 4., 12., 20.,  4., 12., 20.])

grid_cells =
    tensor([[-16., -16.,  24.,  24.],
            [ -8., -16.,  32.,  24.],
            [  0., -16.,  40.,  24.],
            [-16.,  -8.,  24.,  32.],
            [ -8.,  -8.,  32.,  32.],
            [  0.,  -8.,  40.,  32.]])
```

将`grid_cells`绘制在图表如下，即以每个`grid`中心在各level上作`bbox`，覆盖整个`feat_map`，（下图由于`stride, scale, feat_size`选取的原因，导致出现负值）

<img src="NanoDet%20%E4%BB%A3%E7%A0%81%E8%A7%A3%E8%AF%BB.assets/image-20210318171209139.png" alt="image-20210318171209139" style="zoom: 67%;" />

## sample

```python
gfl_head.py -> sample()
```

从分配器返回的分配结果`AssignResult`对象中提取出正、负样本