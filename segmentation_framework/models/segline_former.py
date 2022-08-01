from turtle import forward
import einops
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


def embed_to_cols(features):
    return einops.rearrange(features, 'b c h w -> (b w) h c')
    

def embed_to_rows(features):
    return einops.rearrange(features, 'b c h w -> (b h) w c')


def embed_image(features):
    return einops.rearrange(features, 'b c h w -> b (h w) c')


def _build_encoder(encoder_name):
    if encoder_name == 'efficientnet_v2_s':
        encoder = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)
        return_nodes = {
            'features.1.1.add': 'layer0',
            'features.2.3.add': 'layer1',            
            'features.3.3.add': 'layer2',
            'features.5.8.add': 'layer3',
            'features.6.14.add': 'layer4',
        }
        return create_feature_extractor(encoder, return_nodes=return_nodes)
    else:
        raise ValueError('Unknown encoder name: {}'.format(encoder_name))


class Decoder(torch.nn.Sequential):
    
    def __init__(self, in_channels, out_channels) -> None:
        layers = [
            torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        ]
        super().__init__(*layers)


class SeglineFormer(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _build_encoder('efficientnet_v2_s')
        self.context = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(256, 8, 1024, activation=torch.nn.GELU(), norm_first=True, batch_first=True),
            4
        )
        
    def forward(self, image):
        features = self.encoder(image)
        return self.context(embed_image(features["layer4"])).shape

if __name__ == "__main__":
    print(SeglineFormer()(torch.randn(1, 3, 512, 512)))