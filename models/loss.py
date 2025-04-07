import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
# Recommend
import torch.nn.functional as F


def attention_transform(feat):
    return F.normalize(feat.pow(2).mean(1).view(feat.size(0), -1))


def similarity_transform(feat):
    feat = feat.view(feat.size(0), -1)
    gram = feat @ feat.t()
    return F.normalize(gram)


def SpatialWiseDivergence(feat_t, feat_s):
    assert feat_s.shape[-2:] == feat_t.shape[-2:]  # 确保空间维度相同
    N, C, H, W = feat_s.shape
    # 重新调整张量形状为 (N, H, W, C)，以便在空间位置上应用Softmax
    feat_t_transposed = feat_t.permute(0, 2, 3, 1).reshape(-1, C)
    feat_s_transposed = feat_s.permute(0, 2, 3, 1).reshape(-1, C)

    # 对转置后的特征应用softmax和logsoftmax
    softmax_pred_T = F.softmax(feat_t_transposed / 4.0, dim=1)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    # 计算散度
    loss = torch.sum(softmax_pred_T *
                     logsoftmax(feat_t_transposed / 4.0) -
                     softmax_pred_T *
                     logsoftmax(feat_s_transposed / 4.0)) * (4.0 ** 2)

    # 归一化损失除以总的空间位置数 (N * H * W)
    loss = loss / (N * H * W)

    return loss


def ChannelWiseDivergence(feat_t, feat_s):
    assert feat_s.shape[-2:] == feat_t.shape[-2:]
    N, C, H, W = feat_s.shape
    softmax_pred_T = F.softmax(feat_t.reshape(-1, W * H) / 4.0, dim=1)
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    loss = torch.sum(softmax_pred_T *
                     logsoftmax(feat_t.reshape(-1, W * H) / 4.0) -
                     softmax_pred_T *
                     logsoftmax(feat_s.reshape(-1, W * H) / 4.0)) * (
                   (4.0) ** 2)
    loss = loss / (C * N)
    return loss


class AlignmentLoss(nn.Module):
    def __init__(self,
                 loss_weight=[1.0, 1.0],
                 loss_name='loss_guidance',
                 inter_transform_type='linear'):
        super(AlignmentLoss, self).__init__()
        self.inter_transform_type = inter_transform_type
        self._loss_name = loss_name
        self.loss_weight = loss_weight

    def forward(self, x_guidance_feature):
        loss_inter = x_guidance_feature[0][0].new_tensor(0.0)
        for i in range(2):
            feat_t = x_guidance_feature[0][i]
            feat_s = x_guidance_feature[1][i]
            # print(feat_t.size(),feat_s.size())
            if feat_t.size(-2) != feat_s.size(-2) or feat_t.size(-1) != feat_s.size(-1):
                dsize = (max(feat_t.size(-2), feat_s.size(-2)), max(feat_t.size(-1), feat_s.size(-1)))
                # feat_t = F.interpolate(feat_t, dsize, mode='bilinear', align_corners=False)
                feat_s = F.interpolate(feat_s, dsize, mode='bilinear', align_corners=False)
            loss_inter = loss_inter + self.loss_weight[i] * ChannelWiseDivergence(feat_t, feat_s) + self.loss_weight[
                i] * SpatialWiseDivergence(feat_t, feat_s)
        return loss_inter


class KLDivergenceLoss(nn.Module):
    def __init__(self, loss_weight=[1.0, 1.0], reduction='batchmean'):
        super(KLDivergenceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x_guidance_feature):
        """
        Expect x_guidance_feature to be a tuple of tensors:
        x_guidance_feature[0] -> features from the target distribution
        x_guidance_feature[1] -> features from the source distribution
        """
        loss_kl = 0.0
        for i in range(len(self.loss_weight)):
            feat_t = x_guidance_feature[0][i]
            feat_s = x_guidance_feature[1][i]

            # Ensure both feature maps have the same spatial dimensions
            if feat_t.size(-2) != feat_s.size(-2) or feat_t.size(-1) != feat_s.size(-1):
                dsize = (max(feat_t.size(-2), feat_s.size(-2)), max(feat_t.size(-1), feat_s.size(-1)))
                feat_s = F.interpolate(feat_s, dsize, mode='bilinear', align_corners=False)

            # Calculate KL divergence
            kl_div = F.kl_div(F.log_softmax(feat_t, dim=1), F.softmax(feat_s, dim=1), reduction=self.reduction)
            loss_kl += self.loss_weight[i] * kl_div

        return loss_kl


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=[1.0, 1.0], reduction='mean'):
        super(FeatureConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x_guidance_feature, label_change):
        """
        Expect x_guidance_feature to be a tuple of tensors:
        x_guidance_feature[0] -> features from the target distribution
        x_guidance_feature[1] -> features from the source distribution
        """
        loss_fc = 0.0
        for i in range(len(self.loss_weight)):
            feat_t = x_guidance_feature[0][i]
            feat_s = x_guidance_feature[1][i]

            # Ensure both feature maps have the same spatial dimensions
            if feat_t.size(-2) != feat_s.size(-2) or feat_t.size(-1) != feat_s.size(-1):
                dsize = (max(feat_t.size(-2), feat_s.size(-2)), max(feat_t.size(-1), feat_s.size(-1)))
                feat_s = F.interpolate(feat_s, dsize, mode='bilinear', align_corners=False)

            # Global Channel-wise Max Pooling
            feat_t_max = F.adaptive_max_pool2d(feat_t, 1)
            feat_s_max = F.adaptive_max_pool2d(feat_s, 1)

            # Global Average Pooling
            feat_t_avg = F.adaptive_avg_pool2d(feat_t, 1)
            feat_s_avg = F.adaptive_avg_pool2d(feat_s, 1)

            # Concatenate pooled features
            feat_t_cat = torch.cat((feat_t_max, feat_t_avg), dim=1) * F.interpolate(label_change.float().unsqueeze(1),
                                                                                    size=(feat_t_max.size(2),
                                                                                          feat_t_max.size(3)),
                                                                                    mode='nearest')
            feat_s_cat = torch.cat((feat_s_max, feat_s_avg), dim=1) * F.interpolate(label_change.float().unsqueeze(1),
                                                                                    size=(feat_t_max.size(2),
                                                                                          feat_t_max.size(3)),
                                                                                    mode='nearest')

            # Compute MSE Loss
            mse_loss = F.mse_loss(feat_t_cat, feat_s_cat, reduction=self.reduction)

            # Accumulate loss with weights
            loss_fc += self.loss_weight[i] * mse_loss

        return loss_fc


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    target = F.one_hot(target, num_classes=2)
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def CE_Loss(inputs, target, cls_weights):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights)(temp_inputs, temp_target)
    return CE_loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    # loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss


def weighted_BCE(output, target, weight_pos=None, weight_neg=None):
    output = torch.clamp(output, min=1e-8, max=1 - 1e-8)

    if weight_pos is not None:
        loss = weight_pos * (target * torch.log(output)) + \
               weight_neg * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()

        label_change = F.interpolate(label_change.float().unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
        # print(x1.shape,x2.shape,label_change.shape)
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss


class SCA_Loss(nn.Module):
    def __init__(self):
        super(SCA_Loss, self).__init__()
        self.loss_f = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.alpha = 0.2

    def forward(self, p1, p2, gt_mask):
        b, c, h, w = p1.size()
        p1 = F.softmax(p1, dim=1)
        p2 = F.softmax(p2, dim=1)

        un_gt_mask = 1 - gt_mask
        p1_change = (p1 * gt_mask)
        p2_change = p2 * gt_mask
        p1_unchange = p1 * un_gt_mask
        p2_unchange = p2 * un_gt_mask
        # p1_change = p1_change.permute(0,2,3,1)
        # p2_change = p2_change.permute(0,2,3,1)
        # p1 = torch.reshape(p1,[b*h*w,c])
        # p2 = torch.reshape(p2,[b*h*w,c])  
        # loss = 0.8*self.loss_f(p1_unchange,(nn.Softmax(dim=1)(p2_unchange)).argmax(dim=1))-self.loss_f(p1_change,(nn.Softmax(dim=1)(p2_change)).argmax(dim=1))*0.2
        losses = self.loss_f(p1_change.contiguous().view(b, -1), p2_change.contiguous().view(b, -1)) * (
                    1 - self.alpha) - self.alpha * self.loss_f(p1_unchange.contiguous().view(b, -1),
                                                               p2_unchange.contiguous().view(b, -1))
        loss = torch.mean(losses)
        # print(losses,loss)
        return loss


class ChangeSalience(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.MSELoss(reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)[:, 0, :, :]
        x2 = F.softmax(x2, dim=1)[:, 0, :, :]

        loss = self.loss_f(x1, x2.detach()) + self.loss_f(x2, x1.detach())
        return loss * 0.5


def pix_loss(output, target, pix_weight, ignore_index=None):
    # Calculate log probabilities
    if ignore_index is not None:
        active_pos = 1 - (target == ignore_index).unsqueeze(1).cuda().float()
        pix_weight *= active_pos

    batch_size, _, H, W = output.size()
    logp = F.log_softmax(output, dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W))
    # Multiply with weights
    weighted_logp = (logp * pix_weight).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / pix_weight.view(batch_size, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


# 添加蒸馏损失
class DistillationLoss(nn.Module):
    """用于在线蒸馏的损失函数，支持特征蒸馏和输出蒸馏
    
    参数:
        alpha (float): 软标签蒸馏损失权重
        temperature (float): 温度参数，用于软化概率分布
        reduction (str): 损失的缩减方式，可以是'mean', 'sum', 'batchmean', 'none'
    """
    def __init__(self, alpha=0.5, temperature=4.0, reduction='batchmean'):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        # KL散度损失只支持'none', 'mean', 'sum', 'batchmean'
        self.kl_div = nn.KLDivLoss(reduction=reduction)
    
    def forward(self, student_outputs, teacher_outputs):
        """计算教师网络和学生网络输出之间的KL散度损失
        
        参数:
            student_outputs (Tensor): 学生网络的输出
            teacher_outputs (Tensor): 教师网络的输出
        """
        # 应用温度缩放
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        # 增强对类别1(变化区域)的关注
        # 提取类别1的概率
        student_class1 = soft_student[:, 1:2]
        teacher_class1 = soft_teacher[:, 1:2]
        
        # 计算基本KL散度损失
        loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 为类别1添加额外的MSE损失，更关注变化区域
        # MSE loss只支持'none', 'mean', 'sum'
        actual_reduction = 'mean' if self.reduction == 'batchmean' else self.reduction
        class1_loss = F.mse_loss(
            student_class1.exp(),  # 转回概率
            teacher_class1, 
            reduction=actual_reduction
        ) * 2.0  # 对类别1的权重提高
        
        # 组合损失
        total_loss = loss + class1_loss
        
        return total_loss


class FeatureDistillationLoss(nn.Module):
    """特征蒸馏损失，用于在异源特征间进行知识迁移
    
    参数:
        reduction (str): 损失的缩减方式，可以是'mean', 'sum'
    """
    def __init__(self, reduction='mean'):
        super(FeatureDistillationLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, student_features, teacher_features, feature_mask=None):
        """计算教师网络和学生网络特征之间的MSE损失
        
        参数:
            student_features (Tensor): 学生网络的特征
            teacher_features (Tensor): 教师网络的特征
            feature_mask (Tensor, optional): 特征注意力掩码，用于选择性关注特定特征区域
        """
        # 确保特征尺寸一致
        if student_features.size() != teacher_features.size():
            # 调整学生特征尺寸以匹配教师特征
            student_features = F.interpolate(
                student_features, 
                size=teacher_features.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 默认计算全局损失
        global_loss = F.mse_loss(student_features, teacher_features, reduction=self.reduction)
        
        # 应用特征掩码（如果存在）
        if feature_mask is not None:
            # 确保掩码尺寸与特征匹配
            if feature_mask.size()[2:] != teacher_features.size()[2:]:
                feature_mask = F.interpolate(
                    feature_mask, 
                    size=teacher_features.size()[2:], 
                    mode='nearest'
                )
                
            # 计算局部特征差异
            # 根据掩码权重调整特征学习
            masked_student = student_features * feature_mask
            masked_teacher = teacher_features * feature_mask
            
            # 局部区域损失
            local_loss = F.mse_loss(masked_student, masked_teacher, reduction=self.reduction)
            
            # 对特征图进行通道注意力，增强对重要通道的学习
            student_channel_attention = torch.mean(student_features, dim=[2, 3], keepdim=True)
            teacher_channel_attention = torch.mean(teacher_features, dim=[2, 3], keepdim=True)
            
            # 通道注意力损失
            channel_loss = F.mse_loss(
                student_channel_attention, 
                teacher_channel_attention, 
                reduction=self.reduction
            )
            
            # 组合损失
            # 全局损失权重较低，局部和通道损失权重较高
            loss = global_loss * 0.3 + local_loss * 0.5 + channel_loss * 0.2
        else:
            loss = global_loss
            
        return loss


class MultiLevelDistillationLoss(nn.Module):
    """多层次蒸馏损失，结合特征蒸馏和输出蒸馏
    
    参数:
        feature_weight (float): 特征蒸馏损失权重
        output_weight (float): 输出蒸馏损失权重
        temperature (float): 温度参数，用于软化概率分布
        reduction (str): 损失的缩减方式，可以是'mean', 'sum', 'batchmean', 'none'
    """
    def __init__(self, feature_weight=0.5, output_weight=0.5, temperature=4.0, reduction='batchmean'):
        super(MultiLevelDistillationLoss, self).__init__()
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        self.temperature = temperature
        self.reduction = reduction
        
        self.feature_loss = FeatureDistillationLoss(reduction=self.reduction if self.reduction != 'batchmean' else 'mean')
        self.output_loss = DistillationLoss(alpha=1.0, temperature=temperature, reduction=self.reduction)
        
    def forward(self, student_features, teacher_features, student_outputs, teacher_outputs, feature_mask=None):
        """计算多层次蒸馏损失
        
        参数:
            student_features (Tensor): 学生网络的特征
            teacher_features (Tensor): 教师网络的特征
            student_outputs (Tensor): 学生网络的输出
            teacher_outputs (Tensor): 教师网络的输出
            feature_mask (Tensor, optional): 特征注意力掩码
        """
        # 计算特征蒸馏损失
        feat_loss = self.feature_loss(student_features, teacher_features, feature_mask)
        
        # 计算输出蒸馏损失
        out_loss = self.output_loss(student_outputs, teacher_outputs)
        
        # 结合损失
        total_loss = self.feature_weight * feat_loss + self.output_weight * out_loss
        
        return total_loss, feat_loss, out_loss


class DifferenceAttentionLoss(nn.Module):
    """差异图注意力迁移损失，关注变化区域的特征迁移
    
    参数:
        reduction (str): 损失的缩减方式，可以是'mean', 'sum', 'none'
        alpha (float): 差异图注意力权重
        beta (float): 通道注意力权重
        gamma (float): 空间注意力权重
    """
    def __init__(self, reduction='mean', alpha=1.0, beta=0.5, gamma=0.5):
        super(DifferenceAttentionLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def _generate_difference_map(self, feat1, feat2):
        """生成特征差异图
        
        参数:
            feat1 (Tensor): 第一个特征图
            feat2 (Tensor): 第二个特征图
            
        返回:
            Tensor: 特征差异图
        """
        # 确保特征尺寸一致
        if feat1.size() != feat2.size():
            feat2 = F.interpolate(
                feat2, 
                size=feat1.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 计算差异图 - L2范数
        diff_map = torch.sqrt(torch.sum((feat1 - feat2)**2, dim=1, keepdim=True))
        # 标准化差异图到[0,1]
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        return diff_map
    
    def _generate_channel_attention(self, feat):
        """生成通道注意力图
        
        参数:
            feat (Tensor): 特征图
            
        返回:
            Tensor: 通道注意力图
        """
        # 全局平均池化生成通道描述符
        channel_att = torch.mean(feat, dim=[2, 3], keepdim=True)
        # 标准化
        channel_att = F.softmax(channel_att, dim=1)
        return channel_att
    
    def _generate_spatial_attention(self, feat):
        """生成空间注意力图
        
        参数:
            feat (Tensor): 特征图
            
        返回:
            Tensor: 空间注意力图
        """
        # 跨通道最大池化和平均池化
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        # 合并注意力图
        spatial_att = torch.sigmoid(torch.cat([max_out, avg_out], dim=1).mean(dim=1, keepdim=True))
        return spatial_att
        
    def forward(self, student_features, teacher_features, opt_t1, opt_t2, sar_t2):
        """计算差异图注意力迁移损失
        
        参数:
            student_features (Tensor): 学生网络的特征 (光学t1-SAR异源对)
            teacher_features (Tensor): 教师网络的特征 (光学t1-光学t2同源对)
            opt_t1 (Tensor): 时间点1的光学图像特征
            opt_t2 (Tensor): 时间点2的光学图像特征
            sar_t2 (Tensor): 时间点2的SAR图像特征
            
        返回:
            Tensor: 差异图注意力迁移损失
        """
        # 确保特征尺寸一致
        if student_features.size() != teacher_features.size():
            student_features = F.interpolate(
                student_features, 
                size=teacher_features.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 确保各个输入特征尺寸一致    
        if opt_t1.size()[2:] != teacher_features.size()[2:]:
            opt_t1 = F.interpolate(
                opt_t1, 
                size=teacher_features.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
        if opt_t2.size()[2:] != teacher_features.size()[2:]:
            opt_t2 = F.interpolate(
                opt_t2, 
                size=teacher_features.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
        if sar_t2.size()[2:] != teacher_features.size()[2:]:
            sar_t2 = F.interpolate(
                sar_t2, 
                size=teacher_features.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 1. 生成差异图注意力 - 关注变化区域
        # 教师网络内部差异图 - 光学t1和光学t2的差异
        teacher_diff_map = self._generate_difference_map(opt_t1, opt_t2)
        
        # 学生网络内部差异图 - 光学t1和SAR t2的差异
        student_diff_map = self._generate_difference_map(opt_t1, sar_t2)
        
        # 归一化差异图
        teacher_diff_map = teacher_diff_map / (torch.sum(teacher_diff_map) + 1e-8)
        student_diff_map = student_diff_map / (torch.sum(student_diff_map) + 1e-8)
        
        # 2. 生成通道注意力
        teacher_channel_att = self._generate_channel_attention(teacher_features)
        student_channel_att = self._generate_channel_attention(student_features)
        
        # 3. 生成空间注意力
        teacher_spatial_att = self._generate_spatial_attention(teacher_features)
        student_spatial_att = self._generate_spatial_attention(student_features)
        
        # 4. 计算差异图注意力迁移损失
        diff_att_loss = F.mse_loss(
            student_diff_map * teacher_diff_map, 
            teacher_diff_map,
            reduction=self.reduction
        )
        
        # 5. 计算通道注意力迁移损失
        channel_att_loss = F.mse_loss(
            student_channel_att,
            teacher_channel_att,
            reduction=self.reduction
        )
        
        # 6. 计算空间注意力迁移损失
        spatial_att_loss = F.mse_loss(
            student_spatial_att,
            teacher_spatial_att,
            reduction=self.reduction
        )
        
        # 7. 加权组合损失
        total_loss = self.alpha * diff_att_loss + self.beta * channel_att_loss + self.gamma * spatial_att_loss
        
        return total_loss, diff_att_loss, channel_att_loss, spatial_att_loss
        

class HeterogeneousAttentionDistillationLoss(nn.Module):
    """异源注意力蒸馏损失，结合差异图注意力迁移和特征蒸馏
    
    参数:
        feature_weight (float): 特征蒸馏损失权重
        output_weight (float): 输出蒸馏损失权重
        diff_att_weight (float): 差异图注意力迁移损失权重
        temperature (float): 温度参数，用于软化概率分布
        reduction (str): 损失的缩减方式
    """
    def __init__(self, feature_weight=0.3, output_weight=0.4, diff_att_weight=0.3, 
                 temperature=4.0, reduction='batchmean'):
        super(HeterogeneousAttentionDistillationLoss, self).__init__()
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        self.diff_att_weight = diff_att_weight
        self.temperature = temperature
        self.reduction = reduction
        
        # 特征蒸馏损失
        self.feature_loss = FeatureDistillationLoss(reduction=self.reduction if self.reduction != 'batchmean' else 'mean')
        # 输出蒸馏损失
        self.output_loss = DistillationLoss(alpha=1.0, temperature=temperature, reduction=self.reduction)
        # 差异图注意力迁移损失
        self.diff_att_loss = DifferenceAttentionLoss(reduction='mean')
        
    def forward(self, student_features, teacher_features, student_outputs, teacher_outputs, 
                opt_t1, opt_t2, sar_t2, feature_mask=None):
        """计算异源注意力蒸馏损失
        
        参数:
            student_features (Tensor): 学生网络的特征
            teacher_features (Tensor): 教师网络的特征
            student_outputs (Tensor): 学生网络的输出
            teacher_outputs (Tensor): 教师网络的输出
            opt_t1 (Tensor): 时间点1的光学图像特征
            opt_t2 (Tensor): 时间点2的光学图像特征
            sar_t2 (Tensor): 时间点2的SAR图像特征
            feature_mask (Tensor, optional): 特征注意力掩码
            
        返回:
            tuple: (总损失, 特征损失, 输出损失, 差异图注意力损失)
        """
        # 计算特征蒸馏损失
        feat_loss = self.feature_loss(student_features, teacher_features, feature_mask)
        
        # 计算输出蒸馏损失
        out_loss = self.output_loss(student_outputs, teacher_outputs)
        
        # 计算差异图注意力迁移损失
        diff_att_total, diff_att_loss, channel_att_loss, spatial_att_loss = self.diff_att_loss(
            student_features, teacher_features, opt_t1, opt_t2, sar_t2
        )
        
        # 结合损失
        total_loss = (self.feature_weight * feat_loss + 
                      self.output_weight * out_loss + 
                      self.diff_att_weight * diff_att_total)
        
        return total_loss, feat_loss, out_loss, diff_att_total
