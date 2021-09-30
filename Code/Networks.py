import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Resnet50_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet50_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        if hp.pool_method is not None:
            self.pool_method = eval('nn.' + str(hp.pool_method) + '(1)')
        else:
            self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
            # AdaptiveMaxPool2d, AdaptiveAvgPool2d, AvgPool2d

    def forward(self, input, bb_box = None):
        x = self.features(input)
        if bb_box is None:
            x = self.pool_method(x)
            x = torch.flatten(x, 1)
        elif bb_box is not None:
            x = self.pool_method(x, bb_box)

        return x


class Regulariser(nn.Module):
    def __init__(self):
        super(Regulariser, self).__init__()
        self.params =  torch.nn.Parameter(torch.rand((128, 513), device=device))

    def forward(self, input):
        return (self.params * abs(input)).sum()


class VAE_Network(nn.Module):
    def __init__(self):
        super(VAE_Network, self).__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderCNN()

    # def forward(self, input):
    #     x = self.backbone(input) # B, 512, 8, 8
    #     x = self.pool_method(x).view(-1, 512)
    #     return F.normalize(x)


class EncoderCNN(nn.Module):
    def __init__(self, hp=None):
        super(EncoderCNN, self).__init__()
        self.feature = Unet_Encoder(in_channels=3)
        self.invar = nn.Linear(512, 128)
        # self.var = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(512, 128)
        self.fc_std = nn.Linear(512, 128) # sain 128

    def forward(self, x):
        x = self.feature(x)
        x_invar = self.invar(x)
        # x_var = self.var(x)
        mean = self.fc_mu(x)
        log_var = self.fc_std(x)
        posterior_dist = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        return posterior_dist, F.normalize(x_invar), F.normalize(x)

class DecoderCNN(nn.Module):
    def __init__(self, hp=None):
        super(DecoderCNN, self).__init__()
        self.model = Unet_Decoder(out_channels=3)

    def forward(self, x):
        return self.model(x)


class Unet_Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet_Encoder, self).__init__()

        self.down_1 = Unet_DownBlock(in_channels, 32, normalize=False)
        self.down_2 = Unet_DownBlock(32, 64)
        self.down_3 = Unet_DownBlock(64, 128)
        self.down_4 = Unet_DownBlock(128, 256)
        self.down_5 = Unet_DownBlock(256, 256)
        self.linear_encoder = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)
        x = self.down_5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_encoder(x)
        x = self.dropout(x)
        return x


class Unet_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Unet_Decoder, self).__init__()
        self.linear_1 = nn.Linear(128, 8*8*256)
        self.dropout = nn.Dropout(0.5)
        self.deconv_1 = Unet_UpBlock(256, 256)
        self.deconv_2 = Unet_UpBlock(256, 128)
        self.deconv_3 = Unet_UpBlock(128, 64)
        self.deconv_4 = Unet_UpBlock(64, 32)
        self.final_image = nn.Sequential(*[nn.ConvTranspose2d(32, out_channels,
                                        kernel_size=4, stride=2,
                                        padding=1), nn.Tanh()])

    def forward(self, x):
        x = self.linear_1(x)
        x = x.view(-1, 256, 8, 8)
        x = self.dropout(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.final_image(x)
        return x


class Unet_UpBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc):
        super(Unet_UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(outer_nc),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Unet_DownBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, normalize=True):
        super(Unet_DownBlock, self).__init__()
        layers = [nn.Conv2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=True)]
        if normalize:
            layers.append(nn.InstanceNorm2d(outer_nc))
        layers.append(nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input):
        x = self.backbone(input) # B, 512, 8, 8
        x = self.pool_method(x).view(-1, 512)
        # x = x.mean(dim=[2, 3]) # todo : maxpool
        return F.normalize(x)

class InceptionV3_Network(nn.Module):
    def __init__(self, hp):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c

        if hp.pool_method is not None:
            self.pool_method = eval('nn.' + str(hp.pool_method) + '(1)')
        else:
            self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        # AdaptiveMaxPool2d, AdaptiveAvgPool2d, AvgPool2d

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        # N x 2048 x 8 x 8        # Adaptive average pooling
        feature = self.pool_method(backbone_tensor).view(-1, 2048)
        return F.normalize(feature)

