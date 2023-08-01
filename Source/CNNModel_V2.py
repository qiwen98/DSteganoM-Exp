import torch
import torch.nn.functional as F
import torch.nn as nn
from Utils import corrupt
# from torch_geometric_temporal.nn.attention import tsagcn,

torch.manual_seed(0)

SIGMA_O = 0.1
SIGMA_S = 0.1
BETA = 0.1

def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)

class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialP4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalP3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalP4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalP5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)
        out = torch.cat((p4, p5, p6), 1)
        return out


# Hiding Network (5 conv layers)
class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.initialH3 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialH4 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialH5 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, h):
        h1 = self.initialH3(h)
        h2 = self.initialH4(h)
        h3 = self.initialH5(h)
        mid = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)
        mid2 = torch.cat((h4, h5, h6), 1)
        out = self.finalH(mid2)
        return out


# Reveal Network (2 conv layers)
class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.initialR3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialR4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialR5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalR3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalR4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalR5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalR = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)
        mid = torch.cat((r1, r2, r3), 1)
        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)
        mid2 = torch.cat((r4, r5, r6), 1)
        out = self.finalR(mid2)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 1,1,21,24
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 1,1,21,24
        x = torch.cat([avg_out, max_out], dim=1)  # 1,2,21,24
        x = self.conv1(x)  # 1,2,21,24->1,1,21,24
        return self.sigmoid(x)


class ReluPlusCBAMAttentionBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=3):
        super(ReluPlusCBAMAttentionBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 3): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        bn = {1: nn.BatchNorm1d,
              3: nn.BatchNorm2d}[conv_dim]
        self.conv = nn.Sequential(
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            bn(c_out),
            nn.ReLU()
        )

        self.ca = ChannelAttention(c_out)
        self.sa = SpatialAttention()
        self.isDownUpSample = True if c_in != c_out else False

        self.downUpSampleBlock = nn.Sequential(
            nn.Conv2d(c_in, c_out,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        residual = x
        x2 = self.conv(x)  # 1,3,21,24
        x2 = self.ca(x2) * x2  # 1,3,21,24
        x2 = self.sa(x2) * x2  # 1,3,21,24

        if self.isDownUpSample:
            residual = self.downUpSampleBlock(x)

        x2 += residual

        return x2

class SkipGateReluBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=3):
        super(SkipGateReluBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 3): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        bn = {1: nn.BatchNorm1d,
              3: nn.BatchNorm2d}[conv_dim]
        self.conv = nn.Sequential(
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            bn(c_out),
            nn.ReLU()
            )
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.skip = True if c_in == c_out else False


    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        if self.skip: out += x
        return out

class ReluBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=3):
        super(ReluBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 3): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        bn = {1: nn.BatchNorm1d,
              3: nn.BatchNorm2d}[conv_dim]
        self.conv = nn.Sequential(
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            bn(c_out),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

class SkipGatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=3):
        super(SkipGatedBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 3): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.skip = True if c_in == c_out else False

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        if self.skip: out += x
        return out

class BNGatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=3):
        super(BNGatedBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 3): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        bn=nn.BatchNorm2d
        self.gate = nn.Sequential(
            bn(c_in),
            nn.ReLU(),
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            #conv(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            #bn(c_out)

        )

        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.skip = True if c_in == c_out else False
        #self.linear= nn.Linear(c_out,c_out)
        self.layers=[]

        self.layers.append(self.conv)
        self.layers.append(self.gate)
        #self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            init_layer(layer)
            #pass

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        # if self.skip:
        #     out = x2 * x1 + (1 - x2) * x
        # else:
        out =x2*x1

        # can add dropout here
        if self.skip: out += x # can try to add arithemic here
        return out

class AttentiveBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=3):
        super(AttentiveBlock, self).__init__()

        # up conv
        self.conv_up = nn.Conv2d(c_in, c_out, 1, 1)

        # temporal attention
        self.conv_ta = nn.Conv1d(c_out, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        # s attention
        ker_jpt = 3
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(c_out, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(c_out, c_out // rr)
        self.fc2c = nn.Linear(c_out // rr, c_out)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, y):
        # increase the in_channel_size to out channel_size eg, 3->64
        y = self.conv_up(y)

        # spatial attention
        se = y.mean(-1)  # N C V T-> N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-1) + y

        # temporal attention
        se = y.mean(-2)  # N C V T-> N C T
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-2) + y

        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y


class GatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=3):
        super(GatedBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 3): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        return out


class Encoder(nn.Module):
    def __init__(self, conv_dim=3, block_type='normal', n_layers=3):
        super(Encoder, self).__init__()
        block = {'normal': GatedBlock,
                 'Attentive': AttentiveBlock,
                 'skip': SkipGatedBlock,
                 'relu': ReluBlock,
                 'skiprelu': SkipGateReluBlock,
                 'CBAMAttention': ReluPlusCBAMAttentionBlock,
                 'BNGated': BNGatedBlock
                 }[block_type]

        layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]

        for i in range(n_layers - 1):
            layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h


class CarrierDecoder(nn.Module):
    def __init__(self, conv_dim, block_type='normal',dataset="MTM", n_layers=4,is_ours=False):
        super(CarrierDecoder, self).__init__()
        block = {'normal': GatedBlock,
                 'Attentive': AttentiveBlock,
                 'skip': SkipGatedBlock,
                 'relu': ReluBlock,
                 'skiprelu': SkipGateReluBlock,
                 'CBAMAttention': ReluPlusCBAMAttentionBlock,
                 'BNGated': BNGatedBlock
                 }[block_type]

        layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]

        for i in range(n_layers - 2):
            layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

        layers.append(block(c_in=64, c_out=3, kernel_size=1, stride=1, padding=0, deconv=False))

        self.main = nn.Sequential(*layers)
        self.is_ours =is_ours
        self.dataset = dataset
        if self.is_ours:
          if self.dataset=="CMU" or self.dataset=="Combined":
            self.pooling = nn.AdaptiveAvgPool2d((38, 300)) # 38,300
          else:
            
            self.pooling = nn.AdaptiveAvgPool2d((21, 24)) # 21,24

    def forward(self, x):
        if self.is_ours:
            h = self.main(x)
            h = self.pooling(h)
        else:
            h = self.main(x)

        return h


class MsgDecoder(nn.Module):
    def __init__(self, conv_dim=3, block_type='normal',n_layers=6,dataset="MTM",is_ours=False):
        super(MsgDecoder, self).__init__()
        block = {'normal': GatedBlock,
                 'Attentive': AttentiveBlock,
                 'skip': SkipGatedBlock,
                 'relu': ReluBlock,
                 'skiprelu': SkipGateReluBlock,
                 'CBAMAttention': ReluPlusCBAMAttentionBlock,
                 'BNGated': BNGatedBlock
                 }[block_type]

        # self.main = nn.Sequential(
        #     block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
        #     block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
        #     block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
        #     block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
        #     block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
        #     block(c_in=64, c_out=3, kernel_size=3, stride=1, padding=1, deconv=False)
        # )

        layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]

        for i in range(n_layers - 2):
            layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

        layers.append(block(c_in=64, c_out=3, kernel_size=3, stride=1, padding=1, deconv=False))

        self.main = nn.Sequential(*layers)
        self.is_ours =is_ours
        self.dataset = dataset
        if self.is_ours:
            if self.dataset=="CMU":
                self.pooling = nn.AdaptiveAvgPool2d((38, 300)) # 38,300
            else:
                ## combined or MTM
                self.pooling = nn.AdaptiveAvgPool2d((21, 24)) # 21,24

    def forward(self, x):
        if self.is_ours:
            h = self.main(x)
            h = self.pooling(h)

        else:
            h = self.main(x)
        return h


class RecurrentGCN(torch.nn.Module):
    def __init__(self, block_type, enc_n_layers, dec_c_n_layers, dec_m_n_layers,is_corrupted,dataset="CMU",sigma_o=0.1):
        super(RecurrentGCN, self).__init__()
        self.block_type = block_type
        self.is_ours=False
        self.dataset=dataset
        if block_type =='OursGated':
          self.block_type='BNGated'
          self.is_ours=True
          print('ours')
        self.enc_n_layers = enc_n_layers
        self.dec_c_n_layers = dec_c_n_layers
        self.dec_c_conv_dim = 3 + 3 + 64
        self.dec_m_conv_dim = 3
        self.dec_m_num_repeat = dec_m_n_layers
        self.is_corrupted = is_corrupted
        self.sigma_o=sigma_o
        self.sigma_s=0.1
        self.beta=0.1

        if self.block_type != 'Baluja':
            self.enc_c = Encoder(block_type=self.block_type,
                                 n_layers=self.enc_n_layers)

            self.dec_c = CarrierDecoder(conv_dim=self.dec_c_conv_dim,
                                        block_type=self.block_type,
                                        n_layers=self.dec_c_n_layers,is_ours=self.is_ours,dataset=self.dataset)

            self.dec_m = MsgDecoder(conv_dim=self.dec_m_conv_dim,
                                    block_type=self.block_type,n_layers=self.dec_m_num_repeat,dataset=self.dataset,is_ours=self.is_ours)
        else:
            self.enc_c = PrepNetwork()

            self.dec_c = HidingNetwork()

            self.dec_m = RevealNetwork()

    def forward(self, carrier, msg):
        N, C, V, T = carrier.size()


        if self.is_ours:
          # upsampling the carrier
          carrier = F.interpolate(carrier, size=(V*2, T*2), mode='bilinear',align_corners=True)
        carrier_enc = self.enc_c(carrier)  # encode the carrier
        # print('carrierencode',carrier_enc.shape)
        msg_enc = msg  # concat all msg_i into single tensor

        # if self.block_type == 'Baluja':
        #     msg_enc = F.interpolate(msg_enc, size=(V, T), mode='bilinear',align_corners=True)
        print('msg_enc',msg_enc.shape)
        if self.is_ours:
          # upsampling the secret
          msg_enc = F.interpolate(msg_enc, size=(V*2, T*2), mode='bilinear',align_corners=True)
        if self.block_type != 'Baluja':
            merged_enc = torch.cat((carrier_enc, carrier, msg_enc), dim=1)  # concat encodings on features axis
        else:
            merged_enc = torch.cat((carrier_enc, msg_enc), dim=1)

        carrier_reconst = self.dec_c(merged_enc)  # decode carrier
        print('carrier_reconst',carrier_reconst.shape)

        ỹ_cover_corrupted = carrier_reconst.permute(0, 3, 2, 1).contiguous().view(N * T, V, C)
        # print("ỹ_cover_corrupted",ỹ_cover_corrupted.shape)
        # do corruption
        ỹ_cover_corrupted = corrupt(ỹ_cover_corrupted, self.sigma_o, self.sigma_s, self.beta)
        # decoder
        # recover from corrupt shape   #N,T,V,C -> N,C,V,T
        x = ỹ_cover_corrupted.unsqueeze(0).permute(0, 3, 2, 1).contiguous()
        if not self.is_corrupted:
          msg_reconst = self.dec_m(carrier_reconst) # change this to 'x' if corrupted else 'carrier_reconst'
          #print('not_corrupted_model')
        else:
          msg_reconst = self.dec_m(x) # change this to 'x' if corrupted else 'carrier_reconst'
          #print('corrupted_model')
        print('msg_reconst',msg_reconst.shape)
        return carrier_reconst, msg_reconst