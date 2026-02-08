import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bn=False, relu=True):
        super(Conv2d, self).__init__()
        padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CLRNet(nn.Module):
    def __init__(self, load_weights=False,block_size=3):
        super(CLRNet, self).__init__()
        self.block_size=block_size
        self.Conv2_2f = [64, 64, 'M', 128, 128]
        self.Conv3_3f = ['M', 256, 256, 256]
        self.Conv4_3f = ['M', 512, 512, 512]
        self.Conv5_3f = ['M', 512, 512, 512]

        self.Conv2_2 = make_layers(self.Conv2_2f, in_channels=3, batch_norm=True)
        self.Conv3_3 = make_layers(self.Conv3_3f, in_channels=128, batch_norm=True)
        self.Conv4_3 = make_layers(self.Conv4_3f, in_channels=256, batch_norm=True)
        self.Conv5_3 = make_layers(self.Conv5_3f, in_channels=512, batch_norm=True)

        self.T1 = nn.Sequential(Conv2d(1025+128, 256, 1, bn=True), Conv2d(256, 256, 3, bn=True))
        self.T2 = nn.Sequential(Conv2d(513+64, 128, 1, bn=True), Conv2d(128, 128, 3, bn=True))
        self.T3 = nn.Sequential(
            Conv2d(257+32, 64, 1, bn=True),
            Conv2d(64, 64, 3, bn=True),
            Conv2d(64, 32, 3, bn=True),
            nn.Conv2d(32, 1, 1),
            nn.BatchNorm2d(1)
        )

        self.d1024b = Conv2d(1024, 128, 1, bn=True, relu=True)
        self.d512b = Conv2d(512, 64, 1, bn=True, relu=True)
        self.d256b = Conv2d(256, 32, 1, bn=True, relu=True)

        self.dsn128 = Conv2d(128, 1, 1, bn=True)
        self.dsn64 = Conv2d(64, 1, 1, bn=True)
        self.dsn32 = Conv2d(32, 1, 1, bn=True)

        self.to_mask = nn.Sequential(nn.Conv2d(9, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

        self.to1 = nn.Conv2d(1, 1, 1)
        self.to2 = nn.Conv2d(1, 1, 1)
        self.to3 = nn.Conv2d(1, 1, 1)
        
        self._initialize_weights()
        self.spcial_weights([self.to1, self.to2, self.to3])

    def spcial_weights(self, ml):
        for m in ml:
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_vgg_feature(self, img):
        c2 = self.Conv2_2(img)
        c3 = self.Conv3_3(c2)
        c4 = self.Conv4_3(c3)
        c5 = self.Conv5_3(c4)
        c5 = F.interpolate(c5, scale_factor=2, mode='bilinear', align_corners=False)
        s1 = torch.cat((c5, c4), 1)
        return c2, c3, s1

    def forward(self, images):
        cur = images['cur']
        ref = images['ref']
        
        c2, c3, s1 = self.get_vgg_feature(torch.cat((ref, cur), dim=0))
        
        v_s1 = self.d1024b(s1)
        r1 = self.dsn128(v_s1)
        s1 = self.across_att(v_s1, s1,block_size=self.block_size ,stage=1)
        
        t1 = self.T1(torch.cat((s1, r1), 1))
        t1 = F.interpolate(t1, scale_factor=2, mode='bilinear', align_corners=False)
        s2 = torch.cat((t1, c3), 1)
        
        v_s2 = self.d512b(s2)
        r2 = self.dsn64(v_s2)
        s2 = self.across_att(v_s2, s2,block_size=self.block_size, stage=2)
        
        t2 = self.T2(torch.cat((s2, r2), 1))
        t2 = F.interpolate(t2, scale_factor=2, mode='bilinear', align_corners=False)
        s3 = torch.cat((t2, c2), 1)

        v_s3 = self.d256b(s3)
        r3 = self.dsn32(v_s3)
        s3 = self.across_att(v_s3, s3, block_size=self.block_size, stage=3)
        
        r4 = self.T3(torch.cat((s3, r3), 1))
        r4 = F.interpolate(r4, scale_factor=2, mode='bilinear', align_corners=False)
        mask = torch.sigmoid(r4)
        r4 = mask * torch.relu(r4)

        return [r4, r3, r2, r1], mask

    def across_att(self, v_feat, s_feat,block_size, stage=3):
        B, C, H, W = v_feat.shape
        B = int(B / 2)
        v_ref, v_cur = torch.split(v_feat, B, dim=0)
        s_ref, s_cur = torch.split(s_feat, B, dim=0)
        
        res_ref = self.feats_based_att(v_ref, v_cur, s_ref,block_size, stage)
        res_cur = self.feats_based_att(v_cur, v_ref, s_cur, block_size, stage)
        return torch.cat((res_ref, res_cur), dim=0)

    def feats_based_att(self, query, keys, cur, block_size,stage):
        weight = self.get_feature_based_att(query, keys, block_size=block_size, stage=stage)
        return torch.cat((cur, weight), 1)

    def get_feature_based_att(self, query, keys, block_size=3, stage=3):
        K = block_size
        B, C, H, W = query.shape
        
        q_vec = get_cur(query) 
        k_vec = get_kn(keys, k=K) 
        v_vec = k_vec 

        weight = F.cosine_similarity(q_vec, k_vec, dim=-1) 
        weight_mask = weight.transpose(1, 2).contiguous().view(B, K*K, H, W)
        weight_mask = self.to_mask(weight_mask) 

        weight = weight.view(B, -1, H * W, K * K) 
        if stage == 1: weight = self.to1(weight)
        elif stage == 2: weight = self.to2(weight)
        else: weight = self.to3(weight)

        weight = F.softmax(weight.transpose(1, 2).contiguous(), dim=-1) 
        weight = torch.matmul(weight, v_vec).squeeze(dim=2) 
        weight = weight.transpose(1, 2).contiguous().view(B, -1, H, W)
        
        return weight * weight_mask

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_kn(input, k=3, stride=1, dilation=1):
    B, C, H, W = input.shape
    padding = int(dilation * (k - 1) / 2)
    output = F.unfold(input, kernel_size=(k, k), dilation=dilation, padding=padding, stride=stride)
    return output.view(B, C, k*k, -1).permute(0, 3, 2, 1).contiguous()

def get_cur(cur):
    B, C, H, W = cur.shape
    return cur.view(B, C, 1, H * W).transpose(1, 3).contiguous()

# --- MAIN FUNCTION ---
if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = CLRNet(load_weights=False).to(device)
    
    # Simulate two video frames: Current (cur) and Reference (ref)
    # Resolution 384x640 as suggested in your comments
    dummy_cur = torch.randn(1, 3, 512, 512).to(device)
    dummy_ref = torch.randn(1, 3, 512, 512).to(device)
    
    # Prepare input dictionary
    sample_input = {
        'cur': dummy_cur,
        'ref': dummy_ref
    }
    
    # Forward pass
    with torch.no_grad():
        outputs, mask = model(sample_input)
        print (outputs[0].shape)