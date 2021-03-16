import torch
import numpy as np


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.feature_extractor = torch.nn.ModuleList()
        layer1 = torch.nn.Sequential(
            # Resolution [38x38]
            torch.nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ))
        self.feature_extractor.append(layer1)
        layer2 = torch.nn.Sequential(
            # resolution [19x19]
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.feature_extractor.append(layer2)
        # Resolution [9x9]
        layer3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.feature_extractor.append(layer3)
        layer4 = torch.nn.Sequential(
            # Resolution [5x5]
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.feature_extractor.append(layer4)
        layer5 = torch.nn.Sequential(
            # Resolution [3x3]
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.feature_extractor.append(layer5)
        layer6 = torch.nn.Sequential(
            # Resolution [1x1]
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=0
            )
        )
        self.feature_extractor.append(layer6)
        print(torch.cuda.is_available())
        """
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, cfg.MODEL.NUM_CLASSES)
        )"""

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        # TODO: Implement this function (Task  2a)
        out_features = []
        #out_channel = x.shape[0]
        out = x
        for i in range(len(self.feature_extractor)):
            out = self.feature_extractor[i](out)
            out_features.append(out)
        # flatten image
        out_features = np.array(out_features)
        """
        out = out.view(-1, self.num_output_features)
        out = self.classifier(out)"""

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

