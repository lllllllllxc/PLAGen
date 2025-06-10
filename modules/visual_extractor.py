import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)#resnet01

        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)#(8,2048,7,7)patch_feats通常代表从图像中提取的特征图。
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))#(8,2048)最终生成的avg_feats是一个形状为(batch_size,2048)的张量，代表了经过一系列处理后的图像特征。每个元素都是一个经过平均池化和重塑后的图像平均特征向量
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)#这样做通常是为了将特征图从四维张量（批次维度、通道维度、高度维度、宽度维度）转换为一个三维张量
        return patch_feats, avg_feats
