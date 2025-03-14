from torch import nn as nn
from torch.nn import functional as F, Sequential

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer, ResidualBlockNoBNWithDropout


@ARCH_REGISTRY.register()
class MSRResNet_details(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        self.out = out

        out = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


@ARCH_REGISTRY.register()
class MSRResNet_details_dropoutlast_channel07(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_dropoutlast_channel07, self).__init__()
        self.upscale = upscale
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBNWithDropout, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def dropoutblock(self, x, p=0.7):
        return x * p + self.dropout(x) * (1.0 - p)

    def forward(self, x, plist=None):
        if plist is None:
            plist = [1 for i in range(17)]
        feat = self.lrelu(self.conv_first(x))
        feat = self.body1(feat, p=plist[0])
        feat = self.body2(feat, p=plist[1])
        feat = self.body3(feat, p=plist[2])
        feat = self.body4(feat, p=plist[3])
        feat = self.body5(feat, p=plist[4])
        feat = self.body6(feat, p=plist[5])
        feat = self.body7(feat, p=plist[6])
        feat = self.body8(feat, p=plist[7])
        feat = self.body9(feat, p=plist[8])
        feat = self.body10(feat, p=plist[9])
        feat = self.body11(feat, p=plist[10])
        feat = self.body12(feat, p=plist[11])
        feat = self.body13(feat, p=plist[12])
        feat = self.body14(feat, p=plist[13])
        feat = self.body15(feat, p=plist[14])
        out = self.body16(feat, p=plist[15])

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        out = self.dropoutblock(out, p=plist[16])
        out = self.conv_last(out)

        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


@ARCH_REGISTRY.register()
class MSRResNet_details_dropoutlast_element07(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_details_dropoutlast_element07, self).__init__()
        self.upscale = upscale
        self.dropout = nn.Dropout(p=0.7)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))

        feat = self.body1(feat)
        feat = self.body2(feat)
        feat = self.body3(feat)
        feat = self.body4(feat)
        feat = self.body5(feat)
        feat = self.body6(feat)
        feat = self.body7(feat)
        feat = self.body8(feat)
        feat = self.body9(feat)
        feat = self.body10(feat)
        feat = self.body11(feat)
        feat = self.body12(feat)
        feat = self.body13(feat)
        feat = self.body14(feat)
        feat = self.body15(feat)
        out = self.body16(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.lrelu(self.conv_hr(out))
        out = self.dropout(out)
        out = self.conv_last(out)

        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


@ARCH_REGISTRY.register()
class MSRResNet_dropoutlast_channel07(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_dropoutlast_channel07, self).__init__()

        self.upscale = upscale
        self.dropout = nn.Dropout2d(p=0.7)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights([self.upconv1, self.upconv2], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.conv_hr(out))

        out = self.dropout(out)

        out = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


@ARCH_REGISTRY.register()
class MSRResNet_dropoutlast_element07(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_dropoutlast_element07, self).__init__()

        self.upscale = upscale
        self.dropout = nn.Dropout(p=0.7)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights([self.upconv1, self.upconv2], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.dropout(out)

        out = self.conv_last(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
