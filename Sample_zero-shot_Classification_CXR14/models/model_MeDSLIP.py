# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
from sklearn.metrics import log_loss
import torch.nn as nn
import torch
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel

"""
args.N
args.d_model
args.res_base_model
args.H
args.num_queries
args.dropout
args.attribute_set_size
"""


class MeDSLIP(nn.Module):
    def __init__(self, config, pathology_book):
        super(MeDSLIP, self).__init__()
        self.d_model = config["d_model"]
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(
                config["text_encoder"], freeze_layers=None
            ).to(pathology_book["input_ids"].device)
            self.pathology_book = bert_model(
                input_ids=pathology_book["input_ids"],
                attention_mask=pathology_book["attention_mask"],
            )
            self.pathology_book = self.pathology_book.last_hidden_state[:, 0, :]
        self.pathology_embedding_layer = nn.Linear(768, 256)
        self.cl_fc_pathology = nn.Linear(256, 768)
        self.cl_fc_anatomy = nn.Linear(256, 768)

        """ visual backbone"""
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
        }
        resnet = self._get_res_basemodel(config["res_base_model"])
        num_ftrs = int(resnet.fc.in_features / 2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1_anatomy = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2_anatomy = nn.Linear(num_ftrs, self.d_model)
        self.res_l1_pathology = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2_pathology = nn.Linear(num_ftrs, self.d_model)

        self.mask_generator = nn.Linear(num_ftrs, num_ftrs)

        ###################################
        """ Query Decoder"""
        ###################################

        self.H = config["H"]
        decoder_layer = TransformerDecoderLayer(
            self.d_model, config["H"], 1024, 0.1, "relu", normalize_before=True
        )
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder_anatomy = TransformerDecoder(
            decoder_layer, config["N"], decoder_norm, return_intermediate=False
        )
        self.decoder_pathology = TransformerDecoder(
            decoder_layer, config["N"], decoder_norm, return_intermediate=False
        )

        # Learnable Queries
        self.dropout_feas_anatomy = nn.Dropout(config["dropout"])
        self.dropout_feas_pathology = nn.Dropout(config["dropout"])

        # Attribute classifier
        self.classifier_anatomy = nn.Linear(self.d_model, config["attribute_set_size"])
        self.classifier_pathology = nn.Linear(self.d_model, config["attribute_set_size"])

        self.apply(self._init_weights)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"
            )

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)
            print("text feature extractor:", bert_model_name)
        except:
            raise (
                "Invalid model name. Check the config file and pass a BERT model from transformers lybrary"
            )

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def image_encoder(self, xis):
        # patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        batch_size = xis.shape[0]
        res_fea = self.res_features(xis)  # batch_size,feature_size,patch_num,patch_num
        res_fea = rearrange(res_fea, "b d n1 n2 -> b (n1 n2) d")
        x = rearrange(res_fea, "b n d -> (b n) d")
        masks = self.mask_generator(x)
        x_pathology = masks * x
        x_anatomy = (1 - masks) * x
        x_pathology = self.res_l1_pathology(x_pathology)
        x_anatomy = self.res_l1_anatomy(x_anatomy)
        x_pathology = F.relu(x_pathology)
        x_anatomy = F.relu(x_anatomy)

        x_pathology = self.res_l2_pathology(x_pathology)
        x_anatomy = self.res_l2_anatomy(x_anatomy)

        out_emb_pathology = rearrange(x_pathology, "(b n) d -> b n d", b=batch_size)
        out_emb_anatomy = rearrange(x_anatomy, "(b n) d -> b n d", b=batch_size)
        return out_emb_pathology, out_emb_anatomy

    def forward(self, images):
        B = images.shape[0]

        device = images.device
        """ Visual Backbone """
        x, _ = self.image_encoder(images)  # batch_size,patch_num,dim
        features = x.transpose(0, 1)  # patch_num b dim

        query_embed = self.pathology_embedding_layer(self.pathology_book)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        features, ws = self.decoder_pathology(
            query_embed,
            features,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )
        features = self.dropout_feas_pathology(features)
        x = self.classifier_pathology(features).transpose(0, 1)  # B query Atributes

        return x

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
