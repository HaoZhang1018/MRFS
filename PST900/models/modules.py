import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math


#########################################################################################  
###########  Interactive Gated Mix Attention module for Visual Completion ###############
######################################################################################### 

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 2 // reduction, self.dim * 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2) #B  2C
        max_v = self.max_pool(x).view(B, self.dim * 2)
        
        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1)
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1)
        
        Stat_out = self.sigmoid(avg_se+max_se).view(B, self.dim * 2, 1)
        channel_weights = Stat_out.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super(SpatialAttention, self).__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(4, 4*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4*reduction, 2, kernel_size), 
                    nn.Sigmoid())
    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True) #B  1  H  W
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)  #B  1  H  W
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True) #B  1  H  W
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)  #B  1  H  W                
        x_cat = torch.cat((x1_mean_out, x1_max_out,x2_mean_out,x2_max_out), dim=1) # B 4 H W
        spatial_weights = self.mlp(x_cat).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class MixAttention(nn.Module):
    def __init__(self, dim, reduction=1):
        super(MixAttention, self).__init__()
        self.dim = dim
        self.ca_gate = ChannelAttention(self.dim) 
        self.sa_gate = SpatialAttention(reduction=4)

    def forward(self, x1,x2):
        ca_out = self.ca_gate(x1,x2) # 2 B C 1 1
        sa_out = self.sa_gate(x1,x2)  # 2 B 1 H W
        mixatt_out = ca_out.mul(sa_out)  # 2 B C H W
        return mixatt_out


#### Interactive Gated Mix Attention module for Visual Completion ####       
class IGMAVC(nn.Module):
    def __init__(self, dim, reduction=4):
        super(IGMAVC, self).__init__()                         
        #self.gate = nn.Linear(2*dim, 2*dim)
        self.MA = MixAttention(dim)
        self.gate = nn.Sequential(
                    nn.Linear(dim * 2, dim * 2 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim * 2 // reduction, dim),
                    nn.Sigmoid())
        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"
        x1_flat = x1.flatten(2).transpose(1, 2)  ##B HXW C
        x2_flat = x2.flatten(2).transpose(1, 2)  ##B HXW C
        
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))  ##B HXW C
        gated_weight = gated_weight.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2).contiguous() # B C H W
        
        mix_map = self.MA(x1,x2) #2 B C H W 
        
        Gated_attention_x1 = gated_weight*mix_map[0]
        Gated_attention_x2 = (1-gated_weight)*mix_map[1]
                
        out_x1 = x1 + Gated_attention_x2 * x2  # B C H W
        out_x2 = x2 + Gated_attention_x1 * x1  # B C H W
               
        return out_x1, out_x2  


#########################################################################################  
###########  Progressive Cycle Attention module for Semantic Completion ################# 
######################################################################################### 

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) 
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        #kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_atten+ x) 
        x_out = self.proj_drop(x_out)

        return x_out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.sr_ratio = sr_ratio
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)                    
        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x1, x2, H, W):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        assert B1 == B2 and C1 == C2 and N1 == N2, "x1 and x2 should have the same dimensions" 

        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q1 = self.q1(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3) 

        if self.sr_ratio > 1:
            x2_ = x2.permute(0, 2, 1).reshape(B2, C2, H, W) 
            x2_ = self.sr(x2_).reshape(B2, C2, -1).permute(0, 2, 1) 
            x2_ = self.norm(x2_)
            kv2 = self.kv2(x2_).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
        else:
            kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
    
        #kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4) 
        k2, v2 = kv2[0], kv2[1]

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v2).transpose(1, 2).reshape(B2, N2, C2)
        x_out = self.proj(x_atten+x1)
        x_out = self.proj_drop(x_out)

        return x_out

###########  Progressive Cycle Attention module for Smantic Completion ##############################     
class PCASC(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(PCASC, self).__init__()                         
        self.SA_x1 = SelfAttention(dim, num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CA_x1toX2 = CrossAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.SA_x2 = SelfAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CA_x2toX1 = CrossAttention(dim,num_heads=num_heads,qkv_bias=False, qk_scale=None, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)
              
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        assert B1 == B2 and C1 == C2 and H1 == H2 and W1 == W2, "x1 and x2 should have the same dimensions"
        x1_flat = x1.flatten(2).transpose(1, 2)  ##B HXW C
        x2_flat = x2.flatten(2).transpose(1, 2)  ##B HXW C
        x1_self_enhance =  self.SA_x1(x1_flat,H1, W1)
        x2_cross_enhance = self.CA_x1toX2(x2_flat,x1_self_enhance,H1, W1)
        x2_self_enhance = self.SA_x2(x2_cross_enhance,H1, W1)
        x1_cross_enhance = self.CA_x2toX1(x1_self_enhance,x2_self_enhance,H1, W1)  ##B HXW C
        Fuse = self.proj(x1_cross_enhance)   ##B HXW C
        #Fuse = self.proj_drop(Fuse)

        Fuse_out = Fuse.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()
          
        return Fuse_out  