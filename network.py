
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

class HomographyRegression(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        # self.pre_norm = LayerNorm(576)
        encoder_layer = nn.TransformerEncoderLayer(d_model=576, nhead=12, dim_feedforward=2048, activation="gelu", dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=3)
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

def compute_similarity(features_a, features_b):
    b, c, h, w = features_a.shape
    features_a = features_a.transpose(2, 3).contiguous().view(b, c, h*w)
    features_b = features_b.view(b, c, h*w).transpose(1, 2)
    features_mul = torch.bmm(features_b, features_a)
    correlation_tensor = features_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
    correlation_tensor = F.normalize(F.relu(correlation_tensor), p=2, dim=1)
    return correlation_tensor

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
