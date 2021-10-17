import numpy as np
import paddle
import paddle.nn as nn


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = paddle.min(paddle.unsqueeze(a[:, 2], dim=1), b[:, 2]) - paddle.max(paddle.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = paddle.min(paddle.unsqueeze(a[:, 3], dim=1), b[:, 3]) - paddle.max(paddle.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = paddle.clip(iw, min=0)
    ih = paddle.clip(ih, min=0)

    ua = paddle.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = paddle.clip(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Layer):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = paddle.clip(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if paddle.fluid.is_compiled_with_cuda():
                    alpha_factor = paddle.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

                    bce = -(paddle.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(paddle.to_tensor(0).float().cuda())

                else:
                    alpha_factor = paddle.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

                    bce = -(paddle.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(paddle.to_tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = paddle.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = paddle.ones(classification.shape) * -1

            targets[paddle.less_than(IoU_max, 0.4), :] = 0

            positive_indices = paddle.greater_equal(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if paddle.fluid.is_compiled_with_cuda():
                alpha_factor = paddle.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = paddle.ones(targets.shape) * alpha

            alpha_factor = paddle.where(paddle.equal(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = paddle.where(paddle.equal(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

            bce = -(targets * paddle.log(classification) + (1.0 - targets) * paddle.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if paddle.fluid.is_compiled_with_cuda():
                cls_loss = paddle.where(paddle.not_equal(targets, -1.0), cls_loss, paddle.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = paddle.where(paddle.net_equal(targets, -1.0), cls_loss, paddle.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / paddle.clip(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = paddle.clip(gt_widths, min=1)
                gt_heights = paddle.clip(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = paddle.log(gt_widths / anchor_widths_pi)
                targets_dh = paddle.log(gt_heights / anchor_heights_pi)

                targets = paddle.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if paddle.fluid.is_compiled_with_cuda():
                    targets = targets / paddle.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / paddle.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = paddle.abs(targets - regression[positive_indices, :])

                regression_loss = paddle.where(
                    paddle.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * paddle.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(paddle.to_tensor(0).float())

        return paddle.stack(classification_losses).mean(dim=0, keepdim=True), paddle.stack(regression_losses).mean(
            dim=0, keepdim=True)

