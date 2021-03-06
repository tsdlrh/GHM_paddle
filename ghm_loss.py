import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = paddle.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_labels_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels
    )
    return bin_labels, bin_labels_weights

#GHM_C Loss 损失函数，将GHM思想结合分类的交叉熵损失函数
class GHMC(nn.Layer):
    """
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """    
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """ Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if paddle.to_tensor(pred).dim() != paddle.to_tensor(target).dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target, label_weight
        edges = self.edges
        mmt = self.momentum
        weights = paddle.zeros_like(pred)

        # gradient length
        g = paddle.abs(paddle.nn.functional.sigmoid(pred).detach() - target)

        valid = label_weight > 0
        tot = max(valid.sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]).logical_and((g < edges[i + 1]).logical_and(paddle.to_tensor(valid)))
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights = tot / self.acc_sum[i]
                else:
                    weights = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, paddle.to_tensor(target), paddle.to_tensor(weights), reduction='sum') / tot
        return loss * self.loss_weight

#GHM_R Loss损失函数，将GHM思想用于回归的Smooth L1损失函数
class GHMR(nn.Layer):
    """
    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """    
    def __init__(
            self,
            mu=0.02,
            bins=10,
            momentum=0,
            loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        """   
        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = paddle.sqrt(paddle.to_tensor(diff * diff + mu * mu)) - mu

        # gradient length
        g = paddle.abs(paddle.to_tensor(diff) / paddle.sqrt(paddle.to_tensor(mu * mu + diff * diff))).detach()
        weights = paddle.zeros_like(g)

        valid = label_weight > 0

        tot = max(label_weight.sum().item(), 1.0)

        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = ((g >= edges[i]).logical_and((g < edges[i + 1]).logical_and(paddle.to_tensor(valid)))).astype(int)

            num_in_bin = inds.sum().item()

            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights = tot / self.acc_sum[i]

                else:
                    weights = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight

#函数测试
# if __name__=='__main__':
#     ghm=GHMR(bins=10,momentum=0.75)
#     input1=np.array([[[0.025, 0.35], [0.45, 0.85]]]).astype("float32")
#     target_1=np.array([[[1.0, 1.0], [0.0, 1.0]]]).astype('float32')
#     label_weights=np.array([[[1.0, 1.0], [1.0, 1.0]]]).astype('float32')
#     print("input1.shape=",input1.shape)
#     loss=ghm.forward(input1,target_1,label_weights)
#     print(loss)

#
# if __name__ == '__main__':
#     ghm = GHMC(bins=10, momentum=0.75)
#     input1 = paddle.to_tensor([[[0.025, 0.35], [0.45, 0.85]]])
#     target_1 = paddle.to_tensor([[[1.0, 1.0], [0.0, 1.0]]])
#     label_weights = paddle.to_tensor([[[1.0, 1.0], [1.0, 1.0]]])
#     print("input1.shape=", input1.shape)
#     loss = ghm.forward(input1, target_1, label_weights)
#     print(loss)

