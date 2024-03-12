
"""
This file contains some functions adapted from the following repositories:
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    for the main architecture;
    
    https://github.com/filipradenovic/cnnimageretrieval-pytorch
    for the GeM layer;
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import commons

import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from transformers import DeiTModel
import datetime
from model.cct.cct import cct_14_7x2_384

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class FeaturesExtractor(torch.nn.Module):
    """The FeaturesExtractor is composed of two parts: the backbone encoder and the
    pooling/aggregation layer.
    The pooling/aggregation layer is used only to compute global features.
    """
    def __init__(self, args):
        super().__init__()
        backbone = cct_14_7x2_384(pretrained=True, progress=True)
        backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        for p in backbone.parameters():
            p.requires_grad = False
        for name, child in backbone.classifier.blocks.named_children():
            if int(name) > args.freeze_te:
                for params in child.parameters():
                    params.requires_grad = True
        self.encoder = backbone

        #self.avgpool = nn.AdaptiveAvgPool2d((14, 14))
        self.pool = nn.Sequential(L2Norm(), GeM(), Flatten())
        
    def forward(self, x, f_type="local"):
        x = self.encoder(x)
        if f_type == "local":
            x = x.view(-1,24,24,384)
            x = x.permute(0, 3, 1, 2)
            return F.normalize(x, p=2, dim=1)
        elif f_type == "global":   # global and local
            x = x.view(-1,24,24,384)
            x = x.permute(0, 3, 1, 2)
            return [self.pool(x),F.normalize(x, p=2, dim=1)]
        else:
            raise ValueError(f"Invalid features type: {f_type}")


def compute_similarity(features_a, features_b):
    b, c, h, w = features_a.shape
    features_a = features_a.transpose(2, 3).contiguous().view(b, c, h*w)
    features_b = features_b.view(b, c, h*w).transpose(1, 2)
    features_mul = torch.bmm(features_b, features_a)
    correlation_tensor = features_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
    correlation_tensor = F.normalize(F.relu(correlation_tensor), p=2, dim=1)
    return correlation_tensor

class HomographyRegression(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        # self.pre_norm = LayerNorm(576)
        encoder1 = nn.TransformerEncoderLayer(d_model=576, nhead=12, dim_feedforward=2048, activation="gelu", dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder1, num_layers=3)
        self.encoder2 = nn.TransformerEncoder(encoder1, num_layers=3)
        self.pe = nn.Parameter(torch.zeros(1, 576, 576),requires_grad=True)
        output_dim_last_conv = 576*576
        self.linear = nn.Linear(output_dim_last_conv, output_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x += self.pe
        x = x.permute(2,0,1)
        x = self.encoder(x)
        x1 = x.clone()
        x = self.encoder2(x)
        x = x + x1
        x = x.permute(1,2,0)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x.reshape(B, 8, 2)


class Network(nn.Module):
    """
    Overview of the network: 
    name                 input                                       output
    FeaturesExtractor:   (2B x 3 x H x W)                            (2B x 256 x 15 x 15)
    compute_similarity:  (B x 256 x 15 x 15), (B x 256 x 15 x 15)    (B x 225 x 15 x 15)
    HomographyRegression:(B x 225 x 15 x 15)                         (B x 16)
    """
    
    def __init__(self, features_extractor, homography_regression):
        super().__init__()
        self.features_extractor = features_extractor
        self.homography_regression = homography_regression

    def forward(self, operation, args):
        """Compute a forward pass, which can be of different types.
        This "ugly" step of passing the operation as a string has been adapted
        to allow calling different methods through the Network.forward().
        This is because only Network.forward() works on multiple GPUs when using torch.nn.DataParallel().
        
        Parameters
        ----------
        operation : str, defines the type of forward pass.
        args : contains the tensor(s) on which to apply the operation.
        
        """
        assert operation in ["features_extractor", "similarity", "regression", "similarity_and_regression"]
        if operation == "features_extractor":
            if len(args) == 2:
                tensor_images, features_type = args
                return self.features_extractor(tensor_images, features_type)
            else:
                tensor_images = args
                return self.features_extractor(tensor_images, "local")
        
        elif operation == "similarity":
            tensor_img_1, tensor_img_2 = args
            return self.similarity(tensor_img_1, tensor_img_2)
        
        elif operation == "regression":
            similarity_matrix = args
            return self.regression(similarity_matrix)
        
        elif operation == "similarity_and_regression":
            tensor_img_1, tensor_img_2 = args
            similarity_matrix_1to2, features_1, features_2 = self.similarity(tensor_img_1, tensor_img_2)
            output1=self.regression(similarity_matrix_1to2)
            return output1, features_1, features_2         

    def similarity(self, tensor_img_1, tensor_img_2):
        features_1 = tensor_img_1
        features_2 = tensor_img_2
        similarity_matrix_1to2 = compute_similarity(features_1, features_2)
        return similarity_matrix_1to2, features_1, features_2
    
    def regression(self, similarity_matrix):
        return self.homography_regression(similarity_matrix)