import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from torchvision.models import resnet
import torch
from torchvision import models
from torch import nn
from pytorch_wavelets import DWTForward
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from functools import partial
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class TeLU(nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, input):

        return input * torch.tanh(torch.exp(input))


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )






class Atten(nn.Module):
    """
    Lightweight GAM with adaptive attention and multi-scale features
    """

    def __init__(self, in_channels, rate=4):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // rate, 8)

        self.channel_att = nn.Sequential(
            nn.Linear(in_channels, self.reduced_channels),
            TeLU(),
            nn.Linear(self.reduced_channels, in_channels),
            nn.Sigmoid()
        )


        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, 1),
            nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=1),
            nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=2, dilation=2),
            nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=4, dilation=4),
            nn.Conv2d(self.reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

        # Contextual enhancement
        self.contextual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            TeLU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        channel_att = self.channel_att(x.flatten(2).mean(dim=2)).view(b, c, 1, 1)
        x = x * channel_att

        spatial_att = self.spatial_att(x)
        x = x * spatial_att

        x = self.contextual(x)


        return x + 0.5 * x  


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = TeLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2), out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)

        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * self.act(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')

        return out




class BandProcess(nn.Module):
    def __init__(self, in_channels):
        super(BandProcess, self).__init__()
        encoder_kernel_size = [256, 512, 1024, 2048]
        self.Conv1 = ConvBNAct(in_channels, encoder_kernel_size[0], stride=2)
        self.Conv2 = ConvBNAct(encoder_kernel_size[0], encoder_kernel_size[0], stride=2)

        self.DWT_Conv1 = ConvBNAct(encoder_kernel_size[0] * 3, encoder_kernel_size[1])
        self.DWT_Conv2 = ConvBNAct(encoder_kernel_size[1] * 3, encoder_kernel_size[2])
        self.DWT_Conv3 = ConvBNAct(encoder_kernel_size[2] * 3, encoder_kernel_size[3])


        self.DWT = DWTForward(J=1, mode='zero', wave='haar')


    def forward(self, Band_X):
        Band_X_Conv1 = self.Conv1(Band_X)
        Band_X_Conv2 = self.Conv2(Band_X_Conv1)

        yL1, yH1 = self.DWT(Band_X_Conv2)
        DWT1 = self.Sign_Process(yL1, yH1)
        Band_out1 = self.DWT_Conv1(DWT1)

        yL2, yH2 = self.DWT(Band_out1)
        DWT2 = self.Sign_Process(yL2, yH2)
        Band_out2 = self.DWT_Conv2(DWT2)

        yL3, yH3 = self.DWT(Band_out2)
        DWT3 = self.Sign_Process(yL3, yH3)
        Band_out3 = self.DWT_Conv3(DWT3)

        return Band_X_Conv2, Band_out1, Band_out2, Band_out3


    def Sign_Process(self, yL, yH):
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        band1 = yL + (yL * y_LH)
        band2 = yL + (yL * y_HL)
        band3 = yL + (yL * y_HH)

        all_band = torch.cat([band1, band2, band3], dim=1)
        return all_band



class RPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rpe_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rpe_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.rpe_norm(self.rpe_conv(x))


class Stem(nn.Module):
    def __init__(self, img_dim=3, out_dim=64, rpe=True):
        super(Stem, self).__init__()
        self.conv1 = ConvBNAct(img_dim, out_dim//2, kernel_size=3, stride=2, inplace=True)
        self.conv2 = ConvBNAct(out_dim//2, out_dim, kernel_size=3, stride=2, inplace=True)
        self.rpe = rpe
        if self.rpe:
            self.proj_rpe = RPE(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.rpe:
            x = self.proj_rpe(x)
        return x


class NDVI_Process(nn.Module):
    def __init__(self, dim=64, patch=128, mlp_ratio=4., num_heads=8, drop_rate=0.):
        super(NDVI_Process, self).__init__()
        encoder_kernel_size = [256, 512, 1024, 2048]
        # self.Conv1 = ConvBNAct(dim, encoder_kernel_size[0], stride=2)
        # self.Conv2 = ConvBNAct(encoder_kernel_size[0], encoder_kernel_size[0], stride=2)

        self.stem = Stem(img_dim=dim, out_dim=encoder_kernel_size[0], rpe=True)
        self.attn1 = SS2D(d_model=encoder_kernel_size[0], patch=patch, d_state=16)
        self.Down1 = ConvBNAct(encoder_kernel_size[0], encoder_kernel_size[1], stride=2)
        self.attn2 = SS2D(d_model=encoder_kernel_size[1], patch=patch, d_state=16)
        self.Down2 = ConvBNAct(encoder_kernel_size[1], encoder_kernel_size[2], stride=2)
        self.attn3 = SS2D(d_model=encoder_kernel_size[2], patch=patch, d_state=16)
        self.Down3 = ConvBNAct(encoder_kernel_size[2], encoder_kernel_size[3], stride=2)
        self.attn4 = SS2D(d_model=encoder_kernel_size[3], patch=patch, d_state=16)
        self.drop = nn.Dropout(drop_rate)


    def forward(self, NDVI_X):
        NDVI_X_Conv1 = self.stem(NDVI_X)
        NDVI_Out1 = self.drop(self.attn1(NDVI_X_Conv1) + NDVI_X_Conv1)

        NDVI_X_Conv2 = self.Down1(NDVI_Out1)
        NDVI_Out2 = self.drop(self.attn2(NDVI_X_Conv2) + NDVI_X_Conv2)

        NDVI_X_Conv3 = self.Down2(NDVI_Out2)
        NDVI_Out3 = self.drop(self.attn3(NDVI_X_Conv3) + NDVI_X_Conv3)

        NDVI_X_Conv4 = self.Down3(NDVI_Out3)
        NDVI_Out4 = self.drop(self.attn4(NDVI_X_Conv4) + NDVI_X_Conv4)


        return NDVI_Out1, NDVI_Out2, NDVI_Out3, NDVI_Out4




class MFM(nn.Module):
    def __init__(self, dim, height=3, reduction=8):
        super(MFM, self).__init__()

        self.height = height
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

        self.Linear = nn.Linear(dim, dim)

    def forward(self, RGB_X, Band_X, NDVI_X):
        B, C, H, W = RGB_X.shape

        X1 = RGB_X.clone()
        X2 = Band_X.clone()
        X3 = NDVI_X.clone()

        RGB_weight= self.Linear(X1.view(B, H*W, C)).view(B, C, H, W)
        Band_weight = self.Linear(X2.view(B, H*W, C)).view(B, C, H, W)
        NDVI_weight = self.Linear(X3.reshape(B, H*W, C)).view(B, C, H, W)

        RGB_New = X1 + X1 * Band_weight + X1 * NDVI_weight
        Band_New = X2 + X2 * RGB_weight + X2 * NDVI_weight
        NDVI_New = X3 + X3 * Band_weight + X3 * RGB_weight

        in_feats = torch.cat([RGB_New, Band_New, NDVI_New], dim=1)


        # in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )



class FPN(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=64):
        super().__init__()
        self.pre_conv0 = Conv(encoder_channels[0], decoder_channels, kernel_size=1)
        self.pre_conv1 = Conv(encoder_channels[1], decoder_channels, kernel_size=1)
        self.pre_conv2 = Conv(encoder_channels[2], decoder_channels, kernel_size=1)
        self.pre_conv3 = Conv(encoder_channels[3], decoder_channels, kernel_size=1)

        self.post_conv3 = nn.Sequential(Atten(decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        Atten(decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        Atten(decoder_channels))

        self.post_conv2 = nn.Sequential(Atten(decoder_channels),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        Atten(decoder_channels))

        self.post_conv1 = Atten(decoder_channels)
        self.post_conv0 = Atten(decoder_channels)

    def upsample_add(self, up, x):
        up = F.interpolate(up, x.size()[-2:], mode='nearest')
        up = up + x
        return up

    def forward(self, x0, x1, x2, x3):
        x3 = self.pre_conv3(x3)
        x2 = self.pre_conv2(x2)
        x1 = self.pre_conv1(x1)
        x0 = self.pre_conv0(x0)

        x2 = self.upsample_add(x3, x2)
        x1 = self.upsample_add(x2, x1)
        x0 = self.upsample_add(x1, x0)

        x3 = self.post_conv3(x3)
        x3 = F.interpolate(x3, x0.size()[-2:], mode='bilinear', align_corners=False)

        x2 = self.post_conv2(x2)
        x2 = F.interpolate(x2, x0.size()[-2:], mode='bilinear', align_corners=False)

        x1 = self.post_conv1(x1)
        x1 = F.interpolate(x1, x0.size()[-2:], mode='bilinear', align_corners=False)

        x0 = self.post_conv0(x0)

        x0 = x3 + x2 + x1 + x0

        return x0





class MSMNet(Module):
    def __init__(self, num_classes=2):
        super(MSMNet, self).__init__()
        encoder_kernel_size = [256, 512, 1024, 2048]

        ResNet = models.resnet50(pretrained=False)
        self.RGB_FirstConv = ResNet.conv1
        self.RGB_FirstBN = ResNet.bn1
        self.RGB_Act = ResNet.relu
        self.RGB_MaxPool = ResNet.maxpool
        self.RGB_Encoder1 = ResNet.layer1
        self.RGB_Encoder2 = ResNet.layer2
        self.RGB_Encoder3 = ResNet.layer3
        self.RGB_Encoder4 = ResNet.layer4

        self.BandProcess = BandProcess(1)
        self.NDVIProcess = NDVI_Process(1)
        self.MFM1 = MFM(encoder_kernel_size[0])
        self.MFM2 = MFM(encoder_kernel_size[1])
        self.MFM3 = MFM(encoder_kernel_size[2])
        self.MFM4 = MFM(encoder_kernel_size[3])
        self.FPN = FPN(encoder_channels=encoder_kernel_size, decoder_channels=128)
        self.head = nn.Sequential(ConvBNAct(128, encoder_kernel_size[0]),
                                  nn.Dropout(0.1),
                                  nn.UpsamplingBilinear2d(scale_factor=2),
                                  Conv(encoder_kernel_size[0], num_classes, kernel_size=1))

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, RGB_X, B_X, NDVI_X):
        sz = RGB_X.size()[-2:]
        RGB_X_Conv = self.RGB_FirstConv(RGB_X)
        RGB_X_BN = self.RGB_FirstBN(RGB_X_Conv)
        RGB_X_Act = self.RGB_Act(RGB_X_BN)
        RGB_X_MaxPool = self.RGB_MaxPool(RGB_X_Act)
        RGB_X_Encoder1 = self.RGB_Encoder1(RGB_X_MaxPool)
        Band_Process1, Band_Process2, Band_Process3, Band_Process4 = self.BandProcess(B_X)
        NDVI_Process1, NDVI_Process2, NDVI_Process3, NDVI_Process4 = self.NDVIProcess(NDVI_X)

        Main_Out1 = self.MFM1(RGB_X_Encoder1, Band_Process1, NDVI_Process1)

        RGB_X_Encoder2 = self.RGB_Encoder2(Main_Out1)
        Main_Out2 = self.MFM2(RGB_X_Encoder2, Band_Process2, NDVI_Process2)

        RGB_X_Encoder3 = self.RGB_Encoder3(Main_Out2)
        Main_Out3 = self.MFM3(RGB_X_Encoder3, Band_Process3, NDVI_Process3)

        RGB_X_Encoder4 = self.RGB_Encoder4(Main_Out3)
        Main_Out4 = self.MFM4(RGB_X_Encoder4, Band_Process4, NDVI_Process4)

        Out = self.FPN(Main_Out1, Main_Out2, Main_Out3, Main_Out4)

        out = self.head(Out)

        output = F.interpolate(out, sz, mode='bilinear', align_corners=False)






        return output





