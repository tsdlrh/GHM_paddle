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

#### 简介：单阶段检测在训练时存在正负样本的差距，针对easy和hard样本之间的不同，从梯度的方向考虑解决这个两个问题。提出了梯度协调机制GHM,将GHM的思想嵌入到分类的交叉熵损失和用于回归的Smooth-L1损失中，在COCO数据集上，mAP达到了41.6的优良效果。
#### 背景：
#### GHM思想
#### Loss函数的创新

## 二、论文复现
### （1）模型组网的搭建
### （2）模型损失函数的计算实现
### （3）resnet50的模型对齐
### （4）fpn模型的对齐
