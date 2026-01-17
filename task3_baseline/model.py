import torch
from torch import nn

try:
    from torchvision import models
    try:
        from torchvision.models import ResNet18_Weights
    except ImportError:
        ResNet18_Weights = None
except ImportError:
    models = None
    ResNet18_Weights = None


def _build_resnet18(pretrained):
    if models is None:
        raise ImportError("torchvision is required for the resnet18 backbone.")
    if pretrained:
        if ResNet18_Weights is not None:
            return models.resnet18(weights=ResNet18_Weights.DEFAULT)
        return models.resnet18(pretrained=True)
    if ResNet18_Weights is not None:
        return models.resnet18(weights=None)
    return models.resnet18(pretrained=False)


def _adapt_first_conv(conv1, in_channels):
    if conv1.in_channels == in_channels:
        return conv1
    if in_channels not in (1, 4):
        raise ValueError("resnet18 backbone only supports 1, 3, or 4 channels.")

    new_conv = nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_weight = conv1.weight.mean(dim=1, keepdim=True)
        else:
            extra = conv1.weight.mean(dim=1, keepdim=True)
            new_weight = torch.cat([conv1.weight, extra], dim=1)
        new_conv.weight.copy_(new_weight)
        if conv1.bias is not None:
            new_conv.bias.copy_(conv1.bias)
    return new_conv


class MultiModalSRSNet(nn.Module):
    def __init__(
        self,
        e2_dim,
        srs_dim,
        video_mode="rgbd",
        use_srs_input=False,
        radio_dim=128,
        srs_input_dim=256,
        visual_dim=128,
        backbone="simple",
        pretrained=False,
        freeze_backbone=False,
    ):
        super().__init__()
        self.video_mode = video_mode
        self.use_video = video_mode != "none"
        self.use_srs_input = use_srs_input
        self.backbone = backbone

        if video_mode == "rgbd":
            in_channels = 4
        elif video_mode == "rgb":
            in_channels = 3
        elif video_mode == "disparity":
            in_channels = 1
        else:
            in_channels = 1

        if self.use_video:
            if backbone == "simple":
                self.visual = nn.Sequential(
                    nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                visual_out = 64
            elif backbone == "resnet18":
                resnet = _build_resnet18(pretrained)
                resnet.conv1 = _adapt_first_conv(resnet.conv1, in_channels)
                visual_out = resnet.fc.in_features
                resnet.fc = nn.Identity()
                self.visual = resnet
            else:
                raise ValueError("Unknown backbone: {}".format(backbone))
        else:
            self.visual = None
            visual_out = 0

        if self.use_video and freeze_backbone:
            for param in self.visual.parameters():
                param.requires_grad = False

        self.radio = nn.Sequential(
            nn.Linear(e2_dim, radio_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(radio_dim, radio_dim),
            nn.ReLU(inplace=True),
        )

        if self.use_srs_input:
            self.srs_input_net = nn.Sequential(
                nn.Linear(srs_dim, srs_input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(srs_input_dim, srs_input_dim),
                nn.ReLU(inplace=True),
            )
            srs_out = srs_input_dim
        else:
            self.srs_input_net = None
            srs_out = 0

        self.regressor = nn.Sequential(
            nn.Linear(visual_out + radio_dim + srs_out, visual_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(visual_dim, srs_dim),
        )

    def forward(self, video, e2, srs_input=None):
        features = []
        if self.use_video:
            visual_feat = self.visual(video)
            if visual_feat.ndim > 2:
                visual_feat = visual_feat.flatten(1)
            features.append(visual_feat)
        radio_feat = self.radio(e2)
        features.append(radio_feat)
        if self.use_srs_input:
            if srs_input is None:
                raise ValueError("srs_input is required when use_srs_input is True")
            srs_feat = self.srs_input_net(srs_input)
            features.append(srs_feat)
        merged = torch.cat(features, dim=1)
        return self.regressor(merged)
