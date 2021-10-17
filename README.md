# GHM_paddle

## 项目说明：
### 本项目是针对 Gradient Harmonized Single-stage Detector 论文的复现工作，使用的框架是百度飞桨PaddlePaddle平台.目前完成的工作包括：
#### （1）模型组网的搭建
#### （2）模型损失函数的计算实现
#### （3）resnet50的模型对齐
#### （4）fpn模型的对齐

## 一、论文讲解
#### 论文链接 https://arxiv.org/abs/1811.05181
#### 论文代码 https://github.com/libuyu/GHM_Detection


<img src="https://github.com/tsdlrh/Blog_image/blob/master/1.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/2.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/3.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/4.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/5.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/6.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/7.JPG" width="800px">


#### 简介：单阶段检测在训练时存在正负样本的差距，针对easy和hard样本之间的不同，从梯度的方向考虑解决这个两个问题。提出了梯度协调机制GHM,将GHM的思想嵌入到分类的交叉熵损失和用于回归的Smooth-L1损失中，在COCO数据集上，mAP达到了41.6的优良效果。

#### 背景：单阶段检测器在训练时面临的最大问题就是easy和hard样本，以及positive和negative样本之间的不平衡，easy样本的数量多以及背景样本的出现，影响了检测器的训练，这种问题在二阶段检测中并不存在。相关的研究技术包括，OHEM样本挖掘技术和Focal Loss函数。其中OHEM技术丢弃了大部分样本，训练也不是很高效，Focal Loss函数引入了两个超参数，需要进行大量的实验进行调试，同时Focal Loss是一种静态损失，对数据集的分布并不敏感。本论文中指出，类别不平衡问题主要归结于难度不平衡问题，而难度不平衡问题可以归结于正则化梯度分布（gradient norm）的不平衡,如果一个正样本很容量被分类，则模型从该样本中得到的信息量较少，或者说产生的梯度信息很少。从整体上看，负样本多为easy样本，正样本多为hard样本。因此，两种不平衡可以归结于属性上的不平衡。论文的主要贡献：（1）揭示了单阶段检测器在gradient norm分布方面存在显著不足的基本原理，并且提出了一种新的梯度平衡机制(GHM)来处理这个问题。（2）提出了GHM-C以及GHM-R，它们纠正了不同样本的梯度分布，并且对超参数具有鲁棒性。（3）通过使用GHM，我们可以轻松地训练单阶段检测器，无需任何数据采样策略，并且在COCO基准测试中取得了state-of-the-art的结果。

#### GHM思想：

#### (1) GHM-C Loss

对于一个候选框，它的真实便签为p*∈{0,1}，预测的值为p∈[0,1],采用二元交叉熵损失函数：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/1.JPG" width="200px">
假设x为模型输出，p=sigmodi(x),那么上述的交叉熵损失对于x的导数为：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/2.JPG" width="200px">
那么梯度的模值定义为：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/3.JPG" width="200px">
其中g代表了这个样本的难易程度以及它对整个梯度的贡献。

训练样本的梯度密度函数为：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/4.JPG" width="200px">

其中gk为第k个样本的gradient norm.

g的gradient norm为在以g为中心，长度为ε的区域内的样本数，并且由该区域的有效长度进行归一化。定义梯度密度参数
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/5.JPG" width="200px">

N为样本总数

根据梯度密度参数，可以得到分类问题的损失平衡函数：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/6.JPG" width="200px">


####(1) GHM-R Loss
Smooth L1损失函数为：

Smooth L1关于ti的导数为：

对于所有|d|>δ的样本都具有gradient norm,这就不可能仅仅依靠gradient norm来区分不同属性的样本，为了在回归Loss上应用GHM,将传统的SL1损失函数，改变为ASL1形式

当d很小时，近似为一个方差函数L2 Loss,当d很大时，近似为一个线性损失L1 Loss，具有较好的平滑性，其偏导存在且连续，将GHM应用于回归Loss的结果如下：





## 二、论文复现
### （1）模型组网的搭建
### （2）模型损失函数的计算实现
### （3）resnet50的模型对齐
### （4）fpn模型的对齐
