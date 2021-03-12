from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Flatten, Activation, Interpolate
from decoder import PSPDecoder
from encoder.mobilenet_encoder import MobileNetV1Encoder, MobileNetV2Encoder
from encoder.repvgg_encoder import RepVGGEncoder
import initialization
import torch.utils.model_zoo as model_zoo
import torchvision


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        initialization.initialize_decoder(self.decoder)
        initialization.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialization.initialize_head(self.classification_head)

    def forward(self, x):
        x = self.preupsample(x)
        features = self.encoder(x)
        # only the last layer now. I modified in the farward fuction of both encoder and decoder
        # decoder_output = self.decoder(features)
        decoder_output = self.decoder(*features)  # list of each encoder output

        masks = self.segmentation_head(decoder_output)
#         print(features.shape)

        if self.classification_head is not None:
            labels = self.classification_head(features)
#             print(labels.shape)
            return masks, labels

        return masks

    def predict(self, x):

        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)



class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, height, width, dropout=0.2, activation='relu'):
        self.height = height
        self.width = width

        conv = nn.Conv2d(in_channels=in_channels, out_channels=256,
                         kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        batchnorm = nn.BatchNorm2d(num_features=256)
        activation = Activation(activation)
        dropout = nn.Dropout(p=dropout)
        classification = nn.Conv2d(
            in_channels=256, out_channels=classes, kernel_size=1, stride=1, padding=0)
        interpolate = Interpolate(
            size=(self.height, self.width), mode='bilinear', align_corners=True)
        super().__init__(conv, batchnorm, activation, dropout, classification, interpolate)

    # def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
    #     if pooling not in ("max", "avg"):
    #         raise ValueError(
    #             "Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
    #     pool = nn.AdaptiveAvgPool2d(
    #         1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
    #     flatten = Flatten()
    #     dropout = nn.Dropout(
    #         p=dropout, inplace=True) if dropout else nn.Identity()
    #     linear = nn.Linear(in_channels, classes, bias=True)
    #     activation = Activation(activation)
    #     super().__init__(pool, flatten, dropout, linear, activation)
############################################## For RepVgg #########################################################
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}
g8_map = {l: 8 for l in optional_groupwise_layers}
g16_map = {l: 16 for l in optional_groupwise_layers}
g32_map = {l: 32 for l in optional_groupwise_layers}
####################################################################################################################

class PSPNet(SegmentationModel):
    def __init__(
        self,
        preupsample=False,
        encoder_name="mobilenetv1",
        encoder_weights=False,
        encoder_depth=5,
        psp_out_channels=512,              # PSP out channels after concat not yet final
        psp_use_batchnorm=True,
        psp_dropout=0.2,
        in_channels=3,
        classes=1,
        activation='sigmoid',     # Optional[Union[str, callable]]
        upsampling=8,
        dilated=False,
        # Optional[dict]    # {classes:1, pooling:"avg", dropout:0.2, activation:'sigmoid'}
        aux_params=None,
        deploy=False
    ):
        super().__init__()
        
        self.preupsample = nn.UpsamplingBilinear2d(scale_factor=8) if preupsample else nn.Identity()

        if encoder_name == 'mobilenetv1':
            # n_class and input_size can be any value because we will not use it, we will use only the feature part
            # if we change multiplier   out_channels should change also, espacially the last value
            self.encoder = MobileNetV1Encoder(out_channels=(
                3, 64, 128, 256, 512, 1024), depth=encoder_depth, input_size=320, n_class=classes, multiplier=1)
            if encoder_weights:
                print('MobilenetV1 does not have pretrained weights')
        elif encoder_name == 'mobilenetv2':
            # n_class and input_size can be any value because we will not use it, we will use only the feature part
            # If we change multiplier   out_channels should change also, espacially the last value
            self.encoder = MobileNetV2Encoder(out_channels=(
                3, 16, 24, 32, 96, 1280), depth=5, input_size=224, n_class=classes, multiplier=1)
            if encoder_weights:
                mobile_fea_state_dict = torchvision.models.mobilenet_v2(pretrained=True).features.state_dict()
                model_fea_state_dict = self.encoder.features.state_dict()
                for (old_k, old_p) , (pre_k, pre_p) in zip(model_fea_state_dict.items(), mobile_fea_state_dict.items()):
                    model_fea_state_dict.update({old_k: pre_p})
                self.encoder.features.load_state_dict(model_fea_state_dict, strict=False)
        elif encoder_name == 'repvgg':
            # If we change multiplier   out_channels should change also, espacially the last value
            # Note: diliate canot be used with this type of network
            self.encoder = RepVGGEncoder(out_channels=(3, 64, 64, 128, 256, 1280), depth=5,
                num_blocks=[2, 4, 14, 1], num_classes=classes,
                width_multiplier=[1, 1, 1, 2.5], override_groups_map=g4_map, deploy=deploy)      
            if encoder_weights:
                print('I will add the pretrained weight later')    
        else:
            print(f"No such encoder {encoder_name}")

        if dilated:
            self.encoder.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4]
            )

        else:
            self.encoder.make_stride1(
                stage_list=[4, 5]
            )

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        if aux_params:
            # classes, pooling="avg", dropout=0.2, activation=None
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "psp-{}".format(encoder_name)
        self.initialize()


if __name__ == '__main__':
    input = torch.rand(4, 3, 256, 256).cuda()
    model = PSPNet(encoder_name="repvgg", encoder_weights=False, encoder_depth=5, psp_out_channels=512,
                   psp_use_batchnorm=True, psp_dropout=0.2, in_channels=3, classes=1, activation='sigmoid',
                   upsampling=8, dilated=False, aux_params=None).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())

    # model = MobileNetV2Encoder(out_channels=(3, 64, 128, 256, 512, 1024), depth=5, input_size=320, n_class=1, multiplier=1)
    # model_state_dict = model.features.state_dict()
    # mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    # mobile_state_dict = mobilenet.features.state_dict()
    # print('555')


    # for (old_k, old_p) , (pre_k, pre_p) in zip(model_state_dict.items(), mobile_state_dict.items()):
    #     model_state_dict.update({old_k: pre_p})
    # model.features.load_state_dict(model_state_dict, strict=False)