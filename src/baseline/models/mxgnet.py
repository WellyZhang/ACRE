# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResConv(nn.Module):
    def __init__(self, block, layers, planes, zero_init_residual=False,
                 groups=1, width_per_group=64,in_dim=1, norm_layer=None):
        super(ResConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv_f = nn.Conv2d(planes[3], planes[3], kernel_size = 3, bias=False)
        #self.bnf = norm_layer(planes[3])

        self.layer1 = self._make_layer(block, planes[0], layers[0], stride=1, groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        #self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        #self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        #x = self.conv_f(x)
        #x = self.bnf(x)
        #x = self.relu(x)
        return x


class ResConvReason(nn.Module):
    def __init__(self, block, layers, planes, zero_init_residual=False,
                 groups=1, width_per_group=64,in_dim = 64*3,g_dim = 64, norm_layer=None):
        super(ResConvReason, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=3, padding=1,bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0], stride=2, groups=groups,inplanes = self.inplanes, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups,inplanes = self.inplanes + g_dim, norm_layer=norm_layer)
        #self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        #self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(planes[1], planes[1])
        #self.fc_bn = nn.BatchNorm1d(planes[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, inplanes = 64, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x,g):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = torch.cat([x,g],1)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        #print(x.size())
        #x = self.avgpool(x).squeeze()
        #print(x.size())
        #x = self.relu(self.fc_bn(self.fc(x)))

        return x


class ResConvInfer(nn.Module):
    def __init__(self, block, layers, planes,fc_size, zero_init_residual=False,
                 groups=1, width_per_group=64,in_dim = 64*3, norm_layer=None):
        super(ResConvInfer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=1, padding=1,bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        #self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        #self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        #self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[0], fc_size)
        self.fc_bn = nn.BatchNorm1d(fc_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        #print(x.size())
        x = self.avgpool(x).squeeze()
        #print(x.size())
        x = self.relu(self.fc_bn(self.fc(x)))

        return x


class MXEdge(nn.Module):
    def __init__(self,device='cuda',in_dim = 128, out_dim = 64,T=9,num_mod = 4,mod_dim = 16,mod_out_dim = 8):
        super(MXEdge,self).__init__()
 
        self.num_mod = num_mod
        self.mod_dim = mod_dim
        self.mod_out_dim = mod_out_dim
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.T = T
        self.mod_layer_1 = nn.Linear(self.in_dim,self.mod_dim*self.num_mod)
        self.mod_layer_1_bn = nn.BatchNorm1d(self.mod_dim*self.num_mod)


        #self.module_net = nn.ModuleList()

        #for i in range(self.num_mod):
        #    mod = nn.Sequential(
        #              nn.Linear(self.mod_dim,self.mod_out_dim),
        #              nn.BatchNorm1d(self.mod_out_dim),
        #              nn.ReLU(True),
                 
                      #nn.Linear(48,self.mod_out_dim),
                      #nn.BatchNorm1d(self.mod_out_dim),
                      #nn.ReLU(True)
        #              )
        #    self.module_net.append(mod)
       
        self.m_w_1 = nn.Parameter(torch.rand(self.mod_dim*self.num_mod,self.mod_out_dim*self.num_mod),requires_grad=True)
        self.m_b_1 = nn.Parameter(torch.zeros(self.mod_out_dim*self.num_mod),requires_grad=True)
        self.m_w_1,self.m_b_1 = self.init_w(self.m_w_1,self.m_b_1)
        self.m_w_1_mask = self.create_mask(self.m_w_1)

        self.m_bn_1 = nn.BatchNorm1d(self.mod_out_dim*self.num_mod)
        self.relu = nn.ReLU(True)
                              
                           
        self.mplx_attn = nn.Linear(3*self.num_mod*self.mod_out_dim,3*self.num_mod)

        self.rel_local_fc_1 = nn.Linear(self.num_mod*self.mod_out_dim*2*3,self.out_dim)
        self.rel_local_fc_1_bn = nn.BatchNorm1d(self.out_dim)

    def init_w(self,w,b):
        stdv = 1. / np.sqrt(w.size(1))
        w.data.uniform_(-stdv, stdv)
        b.data.uniform_(-stdv, stdv)
        return w,b

    def create_mask(self,w):
        w_mask = torch.zeros_like(w)
        chunk_0_size = w.size(0)//self.num_mod
        chunk_1_size = w.size(1)//self.num_mod
        #b_mask = torch.zeros_like(b)
        for i in range(self.num_mod):
            w_mask[chunk_0_size*i:chunk_0_size*(i+1),chunk_1_size*i:chunk_1_size*(i+1)]=1
        return nn.Parameter(w_mask,requires_grad=False)

    def linear_func(self,x,w,b,m):
        w = w*m
        #print(x.size(),w.size(),b.size(),m.size())
        o = torch.mm(x,w)
        o = o + b.unsqueeze(0).expand_as(o)
        return o

    def module_net(self,x):
        x = self.linear_func(x,self.m_w_1,self.m_b_1,self.m_w_1_mask)
        x = self.m_bn_1(x)
        x = self.relu(x)
        return x

    def set_summarize(self,x,axis):
        x_sum = torch.sum(x,axis)
        x_mean = torch.mean(x,axis)
        x_max,_ = torch.max(x,axis)

        return torch.cat([x_sum,x_mean,x_max],1)


    def forward(self,fl_02,fl_12):
        fm_02 = F.relu(self.mod_layer_1_bn(self.mod_layer_1(fl_02.view(-1,self.in_dim))))
        fm_12 = F.relu(self.mod_layer_1_bn(self.mod_layer_1(fl_12.view(-1,self.in_dim))))
        #fm_02_split = torch.split(fm_02.view(-1,self.num_mod,self.mod_dim),1,1)
        #fm_12_split = torch.split(fm_12.view(-1,self.num_mod,self.mod_dim),1,1)
        #fm_02 = self.mod_conv1d(fm_02.unsqueeze(1))
        #fm_12 = self.mod_conv1d(fm_12.unsqueeze(1))
        #print(fm_02.size(),fm_12.size())
        #fm_02_list = []
        #fm_12_list = []
        #for i,l in enumerate(self.module_net):
            #print(fm_02_split[i].size(),fm_12_split[i].size())
        #    fm_02_list.append(l(fm_02_split[i].squeeze()))
        #    fm_12_list.append(l(fm_12_split[i].squeeze()))
        fm_02 = self.module_net(fm_02)
        fm_12 = self.module_net(fm_12)


        #print(fm_02.size(),fm_12.size(),self.num_mod,self.mod_dim)
        fm_02_sum = self.set_summarize(fm_02.view(-1,self.T,self.num_mod*self.mod_out_dim),1)
        fm_12_sum = self.set_summarize(fm_12.view(-1,self.T,self.num_mod*self.mod_out_dim),1)
        #print(fm_02_sum.size(),self.num_mod*self.mod_out_dim)
        fm_02_attn = F.sigmoid(self.mplx_attn(fm_02_sum)).unsqueeze(2).repeat(1,1,self.mod_out_dim).view(-1,3*self.num_mod*self.mod_out_dim)
        fm_12_attn = F.sigmoid(self.mplx_attn(fm_12_sum)).unsqueeze(2).repeat(1,1,self.mod_out_dim).view(-1,3*self.num_mod*self.mod_out_dim)
        #print(fm_02.size(),fm_02_attn.size())
        fm_02_sum = fm_02_sum * fm_02_attn
        fm_12_sum = fm_12_sum * fm_12_attn
        #print(fm_02_sum.size(),fm_12_sum.size())
        fm_cat = torch.cat([fm_02_sum,fm_12_sum],1)
        fl = F.relu(self.rel_local_fc_1_bn(self.rel_local_fc_1(fm_cat)))
        return fl


class MXGNet(nn.Module):
    def __init__(self, A=80, B=80, device='cuda',num_fl_1d=5,batch_size=32,beta=10):
        super(MXGNet,self).__init__()
        self.A = A
        self.B = B
        self.batch_size = batch_size
        self.num_fl_1d = num_fl_1d
        self.num_fl = num_fl_1d * num_fl_1d
        self.device = device
        self.beta = beta
        self.rel_size = 64
        self.enc_size = 32
        self.g_size = 24
        self.tag_tensor = None

        self.encoder_conv = ResConv(BasicBlock,[1,1],[32,self.enc_size], in_dim=3)

        self.conv_node = nn.Sequential(
              nn.Conv2d(self.enc_size,self.enc_size,kernel_size=4,stride=2,padding=1),
              nn.BatchNorm2d(self.enc_size),
              nn.ReLU(True),

              nn.Conv2d(self.enc_size,self.enc_size,kernel_size=4,stride=2,padding=1),
              nn.BatchNorm2d(self.enc_size),
              nn.ReLU(True)
              )

        self.moe_layer = MXEdge(in_dim=(self.enc_size+self.num_fl)*2,out_dim=self.g_size,T=self.num_fl)

        self.upsample = nn.Sequential(
              nn.ConvTranspose2d(self.g_size,self.g_size,kernel_size=4,stride=2,padding=1),
              nn.BatchNorm2d(self.g_size),
              nn.ReLU(True)
              )

        self.relation_conv = ResConvReason(BasicBlock,[2,1],[128,self.rel_size],in_dim=self.enc_size*3,g_dim = self.g_size)
        self.infer_conv = ResConvInfer(BasicBlock,[2],[128],256,in_dim=self.rel_size*3)

        self.dropout = nn.Dropout(0.5)
        self.infer_fc = nn.Linear(256,3)
        self.meta_conv = nn.Conv2d(self.rel_size,9,kernel_size=5)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.g_pad = nn.Parameter(torch.rand(self.enc_size), requires_grad=True)
        self.l_pad = nn.Parameter(torch.rand(self.enc_size), requires_grad=True)

    def create_tag(self):
        idx = torch.arange(0,self.num_fl).expand(self.batch_size*10,self.num_fl)
        idx = idx.contiguous().unsqueeze(2)
        tag_tensor = torch.zeros(self.batch_size*10,self.num_fl,self.num_fl).scatter_(2,idx,1).float().to(self.device)        
        return tag_tensor

    def encoder_net(self,x):
        conv_out = self.encoder_conv(x)
        conv_out_p = self.conv_node(conv_out).view(-1,self.enc_size,self.num_fl).permute(0,2,1)
        conv_out_p = torch.cat([conv_out_p,self.tag_tensor[:self.batch_size * 10, :, :]],-1)
        conv_out_rep = conv_out_p.unsqueeze(1)
        conv_out_rep = conv_out_rep.repeat(1,self.num_fl,1,1)
        
        return conv_out,conv_out_rep

    def relation_infer(self,fg0,fg1,fg2,fl0,fl1,fl2):        
        fg_cat = torch.cat([fg0.squeeze(),fg1.squeeze(),fg2.squeeze()],1)
        fl0 = fl0.squeeze()
        fl1 = fl1.squeeze()
        fl2 = fl2.squeeze() 

        fl_02 = torch.cat([fl0,fl2.permute(0,2,1,3)],-1).view(-1,self.enc_size*2+self.num_fl*2)
        fl_12 = torch.cat([fl1,fl2.permute(0,2,1,3)],-1).view(-1,self.enc_size*2+self.num_fl*2)
        fl_sum = self.moe_layer(fl_02,fl_12)
        fl_sum = fl_sum.view(-1,self.num_fl_1d,self.num_fl_1d,self.g_size).permute(0,3,1,2).contiguous()
        fl_sum = self.upsample(fl_sum)
        f_rel = self.relation_conv(fg_cat,fl_sum)

        return f_rel

    def forward(self,x):
        self.batch_size = x.size(0)

        if self.tag_tensor is None:
            self.tag_tensor = self.create_tag()

        f_g,f_l = self.encoder_net(x.view(-1,3,self.A,self.B))
        f_g_list = torch.split(f_g.view(-1,10,self.enc_size,f_g.size(2),f_g.size(3)),1,1)
        f_l_list = torch.split(f_l.view(-1,10,self.num_fl,self.num_fl,self.enc_size+self.num_fl),1,1)

        r_h_1 = self.relation_infer(f_g_list[0],f_g_list[1],f_g_list[2],f_l_list[0],f_l_list[1],f_l_list[2])
        r_h_2= self.relation_infer(f_g_list[3],f_g_list[4],f_g_list[5],f_l_list[3],f_l_list[4],f_l_list[5])

        final_input = []

        f_g_pad = self.g_pad.view(1, 1, -1, 1, 1).expand(self.batch_size, -1, -1, 20, 20)
        f_l_pad = self.l_pad.view(1, -1).expand(self.num_fl, -1)
        f_l_pad = torch.cat((f_l_pad, self.tag_tensor[0, :, :]), dim=-1)
        f_l_pad = f_l_pad.view(1, 1, 1, self.num_fl, -1).expand(self.batch_size, -1, self.num_fl, -1, -1)

        for i in range(6, 10):
            r_h_3 = self.relation_infer(f_g_pad,f_g_pad,f_g_list[i],f_l_pad,f_l_pad,f_l_list[i]) 
            final_input.append(torch.cat((r_h_1, r_h_2, r_h_3), dim=1).unsqueeze(1))
        final_input = torch.cat(final_input, dim=1)

        h_cb = self.infer_conv(final_input.view(-1, self.rel_size * 3, 5, 5)).squeeze()         
        pred = self.infer_fc(self.dropout(h_cb)).view(-1, 4, 3)

        return pred