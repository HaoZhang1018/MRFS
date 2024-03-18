import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger

logger = get_logger()

class PointConv(nn.Module):
    """
    Point convolution block: input: x with size(B C H W); output size (B C1 H W)
    """
    def __init__(self, in_dim=64, out_dim=64,dilation=1, norm_layer=nn.BatchNorm2d):
        super(PointConv, self).__init__()
        self.kernel_size = 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dilation = dilation       
        conv_padding = (self.kernel_size // 2) * self.dilation        
                
        self.pconv = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim, self.kernel_size, padding=conv_padding, dilation = self.dilation),
                norm_layer(self.out_dim),
                nn.ReLU(inplace=True)
                )        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pconv(x) 
        return x


class CNNHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=16,
                 align_corners=False):        
        super(CNNHead, self).__init__()
        self.align_corners = align_corners        
        self.in_channels = in_channels        
        F1_in_channels, F2_in_channels, F3_in_channels, F4_in_channels = self.in_channels
        embedding_dim = embed_dim
                
        self.PointConv_1_rgb = PointConv(in_dim=F1_in_channels, out_dim=embedding_dim)
        self.PointConv_1_ir = PointConv(in_dim=F1_in_channels, out_dim=embedding_dim)
        self.PointConv_2_rgb = PointConv(in_dim=F2_in_channels, out_dim=embedding_dim)
        self.PointConv_2_ir = PointConv(in_dim=F2_in_channels, out_dim=embedding_dim)
        self.PointConv_3_rgb = PointConv(in_dim=F3_in_channels, out_dim=embedding_dim)
        self.PointConv_3_ir = PointConv(in_dim=F3_in_channels, out_dim=embedding_dim)
        self.PointConv_4_rgb = PointConv(in_dim=F4_in_channels, out_dim=embedding_dim)
        self.PointConv_4_ir = PointConv(in_dim=F4_in_channels, out_dim=embedding_dim)
                
        self.CNN_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*8+6, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=embedding_dim, out_channels=3, kernel_size=1),
                            nn.Sigmoid()                                                                                
                            )
       
    def forward(self, inputs, original_input):
        # len=4, 1/4,1/8,1/16,1/32
        F1_rgb, F1_ir, F2_rgb, F2_ir, F3_rgb, F3_ir, F4_rgb, F4_ir = inputs
        input_rgb, input_ir = original_input

        ir_rgb_cat = torch.cat([input_rgb, input_ir], dim=1)

        F1_rgb_c = self.PointConv_1_rgb(F1_rgb)
        F1_rgb_c = F.interpolate(F1_rgb_c, size=input_rgb.size()[2:],mode='bilinear',align_corners=self.align_corners)
                        
        F1_ir_c = self.PointConv_1_ir(F1_ir)
        F1_ir_c = F.interpolate(F1_ir_c, size=input_rgb.size()[2:],mode='bilinear',align_corners=self.align_corners)
                      
        F2_rgb_c = self.PointConv_2_rgb(F2_rgb)
        F2_rgb_c = F.interpolate(F2_rgb_c, size=input_rgb.size()[2:], mode='bilinear',align_corners=self.align_corners)
                      
        F2_ir_c = self.PointConv_2_ir(F2_ir)
        F2_ir_c = F.interpolate(F2_ir_c, size=input_rgb.size()[2:],  mode='bilinear',align_corners=self.align_corners)
               
        F3_rgb_c = self.PointConv_3_rgb(F3_rgb)
        F3_rgb_c = F.interpolate(F3_rgb_c, size=input_rgb.size()[2:],  mode='bilinear',align_corners=self.align_corners)
        
        F3_ir_c = self.PointConv_3_ir(F3_ir)
        F3_ir_c = F.interpolate(F3_ir_c, size=input_rgb.size()[2:], mode='bilinear',align_corners=self.align_corners)
                
        F4_rgb_c = self.PointConv_4_rgb(F4_rgb)
        F4_rgb_c = F.interpolate(F4_rgb_c, size=input_rgb.size()[2:],  mode='bilinear',align_corners=self.align_corners)

        F4_ir_c = self.PointConv_4_ir(F4_ir)
        F4_ir_c = F.interpolate(F4_ir_c, size=input_rgb.size()[2:], mode='bilinear',align_corners=self.align_corners)

        F_1_4_cat = torch.cat([F1_rgb_c, F1_ir_c, F2_rgb_c, F2_ir_c, F3_rgb_c, F3_ir_c, F4_rgb_c, F4_ir_c,ir_rgb_cat], dim=1)
        Fuse = self.CNN_fuse(F_1_4_cat)

        return Fuse