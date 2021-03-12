
import torch.nn as nn
from . import _utils as utils
from .all_mobilenet import MobileNetV1, MobileNetV2


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels for example input channels = 1
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution channels, replce the _out_channels[0] with in_channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple(
                [in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

    def make_stride1(self, stage_list):
        stages = self.get_stages()
        for stage_indx in stage_list:
            utils.adjust_stride(
                module=stages[stage_indx],
            )

##################################################################################################################################################################


class MobileNetV1Encoder(MobileNetV1, EncoderMixin):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:4],
            self.features[4:6],
            self.features[6:8],
            self.features[8:14],
            self.features[14:],
        ]

    def forward(self, x):
        # return self.features(x)

        stages = self.get_stages()
        
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        
        return features

    # def load_state_dict(self, state_dict, **kwargs):
    #     # state_dict.pop("classifier.1.bias")
    #     # state_dict.pop("classifier.1.weight")
    #     super().load_state_dict(state_dict, strict=False, **kwargs)

##################################################################################################################################################################


class MobileNetV2Encoder(MobileNetV2, EncoderMixin):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x):
        # return self.features(x)

        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    # def load_state_dict(self, state_dict, strict=False, **kwargs):
    #     # state_dict.pop("classifier.1.bias")
    #     # state_dict.pop("classifier.1.weight")
    #     super().load_state_dict(state_dict, strict=strict, **kwargs)
