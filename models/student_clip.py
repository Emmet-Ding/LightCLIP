import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, image):
        return self.proj(self.backbone(image))


class MobileNetV3ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, image):
        return self.proj(self.backbone(image))


class TinyTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        context_length: int = 77,
        width: int = 256,
        layers: int = 2,
        heads: int = 4,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln_final = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, text_tokens):
        if text_tokens.size(1) > self.context_length:
            raise ValueError(
                f"text length {text_tokens.size(1)} exceeds context_length {self.context_length}"
            )

        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding[: x.size(1)].unsqueeze(0)
        x = self.transformer(x)
        x = self.ln_final(x)

        eot_pos = text_tokens.argmax(dim=-1)
        x = x[torch.arange(x.size(0), device=x.device), eot_pos]
        return self.proj(x)


class StudentCLIP(nn.Module):
    def __init__(
        self,
        image_encoder: str = "resnet18",
        embed_dim: int = 512,
        vocab_size: int = 49408,
        context_length: int = 77,
        text_width: int = 256,
        text_layers: int = 2,
        text_heads: int = 4,
        pretrained_image: bool = False,
    ):
        super().__init__()

        if image_encoder == "resnet18":
            self.image_encoder = ResNet18ImageEncoder(
                embed_dim=embed_dim, pretrained=pretrained_image
            )
        elif image_encoder == "mobilenetv3":
            self.image_encoder = MobileNetV3ImageEncoder(
                embed_dim=embed_dim, pretrained=pretrained_image
            )
        else:
            raise ValueError(f"Unsupported image_encoder: {image_encoder}")

        self.text_encoder = TinyTextEncoder(
            vocab_size=vocab_size,
            context_length=context_length,
            width=text_width,
            layers=text_layers,
            heads=text_heads,
            embed_dim=embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return F.normalize(self.image_encoder(image), dim=-1)

    def encode_text(self, text_tokens):
        return F.normalize(self.text_encoder(text_tokens), dim=-1)

    def forward(self, image, text_tokens):
        image_feat = self.encode_image(image)
        text_feat = self.encode_text(text_tokens)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * image_feat @ text_feat.t()
        return image_feat, text_feat, logits

