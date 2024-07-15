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
    def __init__(
        self, config, anatomy_book, pathology_book, mode="train",
    ):
        super(MeDSLIP, self).__init__()
        self.mode = mode
        self.d_model = config["d_model"]
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(
                config["text_encoder"], freeze_layers=None
            ).to(anatomy_book["input_ids"].device)
            self.anatomy_book = bert_model(
                input_ids=anatomy_book["input_ids"],
                attention_mask=anatomy_book["attention_mask"],
            )  # (**encoded_inputs)
            self.anatomy_book = self.anatomy_book.last_hidden_state[:, 0, :]
            self.pathology_book = bert_model(
                input_ids=pathology_book["input_ids"],
                attention_mask=pathology_book["attention_mask"],
            )  # (**encoded_inputs)
            self.pathology_book = self.pathology_book.last_hidden_state[:, 0, :]
        self.pathology_embedding_layer = nn.Linear(768, 256)
        self.cl_fc_pathology = nn.Linear(256, 768)

        self.pathology_name = [
            "normal",
            "clear",
            "sharp",
            "sharply",
            "unremarkable",
            "intact",
            "stable",
            "free",
            "effusion",
            "opacity",
            "pneumothorax",
            "edema",
            "atelectasis",
            "tube",
            "consolidation",
            "process",
            "abnormality",
            "enlarge",
            "tip",
            "low",
            "pneumonia",
            "line",
            "congestion",
            "catheter",
            "cardiomegaly",
            "fracture",
            "air",
            "tortuous",
            "lead",
            "pathology",
            "calcification",
            "prominence",
            "device",
            "engorgement",
            "picc",
            "clip",
            "elevation",
            "expand",
            "nodule",
            "wire",
            "fluid",
            "degenerative",
            "pacemaker",
            "thicken",
            "marking",
            "scar",
            "hyperinflate",
            "blunt",
            "loss",
            "widen",
            "coll_eapse",
            "density",
            "emphysema",
            "aerate",
            "mass",
            "crowd",
            "infiltrate",
            "obscure",
            "deformity",
            "hernia",
            "drainage",
            "distention",
            "shift",
            "stent",
            "pressure",
            "lesion",
            "finding",
            "borderline",
            "hardware",
            "dilation",
            "chf",
            "redistribution",
            "aspiration",
            "tail_abnorm_obs",
            "excluded_obs",
        ]

        self.excluded_pathology = [
            "pneumonia",
            "infiltrate",
            "mass",
            "nodule",
            "emphysema",
            "fibrosis",
            "thicken",
            "hernia",
        ]

        self.keep_class_dim_pathology = [
            self.pathology_name.index(i)
            for i in self.pathology_name
            if i not in self.excluded_pathology
        ]
        """ visual backbone"""
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
        }
        resnet = self._get_res_basemodel(config["res_base_model"])
        num_ftrs = int(resnet.fc.in_features / 2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])

        self.res_l1_pathology = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2_pathology = nn.Linear(num_ftrs, self.d_model)

        self.cl_fc_anatomy = nn.Linear(256, 768)
        self.res_l1_anatomy = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2_anatomy = nn.Linear(num_ftrs, self.d_model)

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
        self.classifier_pathology = nn.Linear(
            self.d_model, config["attribute_set_size"]
        )

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

        mask = self.mask_generator(x)
        x_pathology = mask * x
        x_anatomy = (1 - mask) * x

        x_pathology = self.res_l1_pathology(x_pathology)
        x_anatomy = self.res_l1_anatomy(x_anatomy)
        x_pathology = F.relu(x_pathology)
        x_anatomy = F.relu(x_anatomy)

        x_pathology = self.res_l2_pathology(x_pathology)
        x_anatomy = self.res_l2_anatomy(x_anatomy)

        out_emb_pathology = rearrange(x_pathology, "(b n) d -> b n d", b=batch_size)
        out_emb_anatomy = rearrange(x_anatomy, "(b n) d -> b n d", b=batch_size)
        return out_emb_pathology, out_emb_anatomy

    def forward(
        self,
        images,
        labels_pathology=None,
        labels_anatomy=None,
        matrix=None,
        sample_index_pathology=None,
        sample_index_anatomy=None,
        is_train=True,
        text_gen=False,
        no_cl=False,
        exclude_class=False,
    ):

        # labels batch,51,75 binary_label batch,75 sample_index_pathology batch,index
        B = images.shape[0]
        device = images.device
        """ Visual Backbone """
        x_pathology, x_anatomy = self.image_encoder(images)  # batch_size,patch_num,dim

        features_pathology = x_pathology.transpose(0, 1)  # patch_num b dim
        features_anatomy = x_anatomy.transpose(0, 1)  # patch_num b dim

        query_embed_pathology = self.pathology_embedding_layer(self.pathology_book)
        query_embed_anatomy = self.pathology_embedding_layer(self.anatomy_book)
        query_embed_pathology = query_embed_pathology.unsqueeze(1).repeat(1, B, 1)
        query_embed_anatomy = query_embed_anatomy.unsqueeze(1).repeat(1, B, 1)

        features_pathology, ws_pathology = self.decoder_pathology(
            query_embed_pathology,
            features_pathology,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )
        features_anatomy, ws_anatomy = self.decoder_anatomy(
            query_embed_anatomy,
            features_anatomy,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )

        ap_pathology = features_pathology
        ap_anatomy = features_anatomy

        ap_logits = torch.bmm(
            ap_pathology.transpose(0, 1), ap_anatomy.transpose(0, 1).transpose(1, 2)
        ).transpose(
            1, 2
        )  # B, 51, 75
        if text_gen:
            output_logits = ap_logits
        matrix_zero = matrix

        masks = matrix_zero >= 0
        ap_logits = ap_logits[masks]
        matrix_zero = matrix_zero[masks]

        loss_ap = F.binary_cross_entropy_with_logits(
            ap_logits.float(), matrix_zero.float()
        )

        out_pathology = self.dropout_feas_pathology(features_pathology)
        out_anatomy = self.dropout_feas_anatomy(features_anatomy)

        if is_train == True and no_cl == False:

            # get anatomytomy query
            anatomytomy_query = torch.zeros(
                [
                    sample_index_pathology.shape[0],
                    sample_index_pathology.shape[1],
                    sample_index_pathology.shape[2],
                    self.anatomy_book.shape[-1],
                ]
            ).to(
                device
            )  # [128, 75, 8, 768]
            entity_query = torch.zeros(
                [
                    sample_index_anatomy.shape[0],
                    sample_index_anatomy.shape[1],
                    sample_index_anatomy.shape[2],
                    self.pathology_book.shape[-1],
                ]
            ).to(device)

            anatomytomy_query = self.anatomy_book[sample_index_pathology, :] * (
                sample_index_pathology != -1
            ).int().unsqueeze(-1).repeat(
                1, 1, 1, 768
            )  # batch, Q , position_num ,dim [128, 75, 8, 768]
            entity_query = self.pathology_book[sample_index_anatomy, :] * (
                sample_index_anatomy != -1
            ).int().unsqueeze(-1).repeat(1, 1, 1, 768)

            matrix_zero_pathology = matrix
            matrix_zero_anatomy = matrix.transpose(1, 2)
            matrix_zero_pathology[matrix_zero_pathology < 1] = 0
            matrix_zero_anatomy[matrix_zero_anatomy < 1] = 0
            matrix_zero_pathology = matrix_zero_pathology.unsqueeze(3).repeat(
                1, 1, 1, anatomytomy_query.shape[-1]
            )
            matrix_zero_anatomy = matrix_zero_anatomy.unsqueeze(3).repeat(
                1, 1, 1, entity_query.shape[-1]
            )

            anatomy_temp = self.anatomy_book
            pathology_temp = self.pathology_book
            anatomy_temp = anatomy_temp.unsqueeze(0).repeat(
                anatomytomy_query.shape[0], 1, 1
            )
            pathology_temp = pathology_temp.unsqueeze(0).repeat(
                entity_query.shape[0], 1, 1
            )
            anatomy_temp = anatomy_temp.unsqueeze(2).repeat(
                1, 1, anatomytomy_query.shape[1], 1
            )
            pathology_temp = pathology_temp.unsqueeze(2).repeat(
                1, 1, entity_query.shape[1], 1
            )

            posi_matrix_pathology = (matrix_zero_pathology * anatomy_temp).transpose(
                1, 2
            )
            posi_matrix_anatomy = (matrix_zero_anatomy * pathology_temp).transpose(1, 2)

            for i in range(anatomytomy_query.shape[0]):
                for j in range(anatomytomy_query.shape[1]):
                    if (posi_matrix_pathology[i, j] != 0).sum() > 0:
                        num_posi = (
                            torch.nonzero(posi_matrix_pathology[i, j], as_tuple=True)[0]
                            .unique()
                            .shape[0]
                        )
                        assert anatomytomy_query[i, j, 0, :].sum() == 0
                        anatomytomy_query[i, j, 0, :] = (
                            posi_matrix_pathology[i, j, :, :].sum(dim=0) / num_posi
                        )

            for i in range(entity_query.shape[0]):
                for j in range(entity_query.shape[1]):
                    if (posi_matrix_anatomy[i, j] != 0).sum() > 0:
                        num_posi = (
                            torch.nonzero(posi_matrix_anatomy[i, j], as_tuple=True)[0]
                            .unique()
                            .shape[0]
                        )
                        assert entity_query[i, j, 0, :].sum() == 0
                        entity_query[i, j, 0, :] = (
                            posi_matrix_anatomy[i, j, :, :].sum(dim=0) / num_posi
                        )
            # Got anatomytomy query

            # [Q,B,A]
            ll_pathology = out_pathology.transpose(0, 1)  # B Q A
            ll_anatomy = out_anatomy.transpose(0, 1)  # B Q A

            Q_pathology = ll_pathology.shape[1]
            Q_anatomy = ll_anatomy.shape[1]

            ll_pathology = ll_pathology.reshape(
                ll_pathology.shape[0] * ll_pathology.shape[1], -1
            )
            ll_anatomy = ll_anatomy.reshape(
                ll_anatomy.shape[0] * ll_anatomy.shape[1], -1
            )

            ll_pathology = self.cl_fc_pathology(ll_pathology)
            ll_anatomy = self.cl_fc_anatomy(ll_anatomy)

            ll_pathology = ll_pathology.unsqueeze(dim=-1)
            ll_anatomy = ll_anatomy.unsqueeze(dim=-1)

            anatomytomy_query = anatomytomy_query.reshape(B * Q_pathology, 8, 768)
            entity_query = entity_query.reshape(B * Q_anatomy, 8, 768)

            ll_pathology = torch.bmm(
                anatomytomy_query, ll_pathology
            ).squeeze()  # B Q position_num
            ll_anatomy = torch.bmm(
                entity_query, ll_anatomy
            ).squeeze()  # B Q position_num

            cl_labels_pathology = torch.zeros((ll_pathology.shape[0])).to(device)
            cl_labels_anatomy = torch.zeros((ll_anatomy.shape[0])).to(device)

            if exclude_class == True:
                cl_labels_pathology = cl_labels_pathology.reshape(B, Q_pathology)
                cl_labels_anatomy = cl_labels_anatomy.reshape(B, Q_anatomy)

                cl_labels_pathology = cl_labels_pathology[
                    :, self.keep_class_dim_pathology
                ]
                cl_labels_anatomy = cl_labels_anatomy[:, self.keep_class_dim_pathology]

                cl_labels_pathology = cl_labels_pathology.reshape(-1)
                cl_labels_anatomy = cl_labels_anatomy.reshape(-1)

                ll_pathology = ll_pathology.reshape(B, Q_pathology, -1)
                ll_anatomy = ll_anatomy.reshape(B, Q_anatomy, -1)

                ll_pathology = ll_pathology[:, self.keep_class_dim_pathology, :]
                ll_pathology = ll_pathology.reshape(
                    B * (len(self.keep_class_dim_pathology)), -1
                )
                ll_anatomy = ll_anatomy.reshape(B * Q_anatomy, -1)

        x_pathology = self.classifier_pathology(out_pathology).transpose(0, 1)  # []
        x_anatomy = self.classifier_anatomy(out_anatomy).transpose(
            0, 1
        )  # B query Atributes

        if exclude_class == True:
            labels_pathology = labels_pathology[:, self.keep_class_dim_pathology]
            x_pathology = x_pathology[:, self.keep_class_dim_pathology, :]

        labels_pathology = labels_pathology.reshape(-1, 1)
        labels_anatomy = labels_anatomy.reshape(-1, 1)
        logits_pathology = x_pathology.reshape(-1, x_pathology.shape[-1])
        logits_anatomy = x_anatomy.reshape(-1, x_anatomy.shape[-1])
        Mask_pathology = ((labels_pathology != -1) & (labels_pathology != 2)).squeeze()
        Mask_anatomy = ((labels_anatomy != -1) & (labels_anatomy != 2)).squeeze()

        cl_mask_pathology = (labels_pathology == 1).squeeze()
        cl_mask_anatomy = (labels_anatomy == 1).squeeze()
        if is_train == True:
            labels_pathology = labels_pathology[Mask_pathology].long()
            labels_anatomy = labels_anatomy[Mask_anatomy].long()
            logits_pathology = logits_pathology[Mask_pathology]
            logits_anatomy = logits_anatomy[Mask_anatomy]
            loss_ce_pathology = F.cross_entropy(
                logits_pathology, labels_pathology[:, 0]
            )
            loss_ce_anatomy = F.cross_entropy(logits_anatomy, labels_anatomy[:, 0])
            if no_cl == False:
                cl_labels_pathology = cl_labels_pathology[cl_mask_pathology].long()
                cl_labels_anatomy = cl_labels_anatomy[cl_mask_anatomy].long()
                ll_pathology = ll_pathology[cl_mask_pathology]
                ll_anatomy = ll_anatomy[cl_mask_anatomy]
                loss_cl_pathology = F.cross_entropy(ll_pathology, cl_labels_pathology)
                loss_cl_anatomy = F.cross_entropy(ll_anatomy, cl_labels_anatomy)
                loss_ce = loss_ce_pathology + loss_ce_anatomy
                loss_cl = loss_cl_pathology + loss_cl_anatomy
                loss = loss_ce + loss_cl + loss_ap
            else:
                loss_cl = torch.tensor(0)
                loss = loss_ce_pathology + loss_ce_anatomy + loss_ap
        else:
            loss = 0
        if is_train == True:
            if text_gen:
                return (
                    loss,
                    x_pathology,
                    ws_pathology,
                    x_anatomy,
                    ws_anatomy,
                    output_logits,
                )
            else:
                return (
                    loss,
                    loss_ce_pathology,
                    loss_cl_pathology,
                    loss_ce_anatomy,
                    loss_cl_anatomy,
                    loss_ap,
                )
        else:
            return loss, x_pathology, ws_pathology, x_anatomy, ws_anatomy

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
