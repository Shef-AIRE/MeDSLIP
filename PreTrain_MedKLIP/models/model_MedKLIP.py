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


class MedKLIP(nn.Module):
    def __init__(self, config, ana_book, disease_book, mode="train", use_pe_cl=True):
        super(MedKLIP, self).__init__()
        self.use_pe_cl = use_pe_cl
        self.mode = mode
        self.d_model = config["d_model"]
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(
                config["text_encoder"], freeze_layers=None
            ).to(ana_book["input_ids"].device)
            self.ana_book = bert_model(
                input_ids=ana_book["input_ids"],
                attention_mask=ana_book["attention_mask"],
            )  # (**encoded_inputs)
            self.ana_book = self.ana_book.last_hidden_state[:, 0, :]
            self.disease_book = bert_model(
                input_ids=disease_book["input_ids"],
                attention_mask=disease_book["attention_mask"],
            )  # (**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:, 0, :]
        self.disease_embedding_layer = nn.Linear(768, 256)
        self.cl_fc_e = nn.Linear(256, 768)
        self.cl_fc_p = nn.Linear(256, 768)

        self.disease_name = [
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
            "disease",
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

        self.excluded_disease = [
            "pneumonia",
            "infiltrate",
            "mass",
            "nodule",
            "emphysema",
            "fibrosis",
            "thicken",
            "hernia",
        ]

        self.keep_class_dim_e = [
            self.disease_name.index(i)
            for i in self.disease_name
            if i not in self.excluded_disease
        ]
        """ visual backbone"""
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
        }
        resnet = self._get_res_basemodel(config["res_base_model"])
        num_ftrs = int(resnet.fc.in_features / 2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1_p = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2_p = nn.Linear(num_ftrs, self.d_model)
        self.res_l1_e = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2_e = nn.Linear(num_ftrs, self.d_model)

        self.mask_generator = nn.Linear(num_ftrs, num_ftrs)


        ###################################
        """ Query Decoder"""
        ###################################

        self.H = config["H"]
        decoder_layer = TransformerDecoderLayer(
            self.d_model, config["H"], 1024, 0.1, "relu", normalize_before=True
        )
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder_p = TransformerDecoder(
            decoder_layer, config["N"], decoder_norm, return_intermediate=False
        )
        self.decoder_e = TransformerDecoder(
            decoder_layer, config["N"], decoder_norm, return_intermediate=False
        )

        # Learnable Queries
        # self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas_p = nn.Dropout(config["dropout"])
        self.dropout_feas_e = nn.Dropout(config["dropout"])

        # Attribute classifier
        self.classifier_p = nn.Linear(self.d_model, config["attribute_set_size"])
        self.classifier_e = nn.Linear(self.d_model, config["attribute_set_size"])

        # # Class classifier
        # self.cls_classifier = nn.Linear(self.d_model,args.num_classes)

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
            model = AutoModel.from_pretrained(bert_model_name)  # , return_dict=True)
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
        x_e = self.mask_generator(x) * x
        x_p = (1 - self.mask_generator(x)) * x
        # batch_size,num,feature_size
        # h = h.squeeze()
        x_e = self.res_l1_e(x_e)
        x_p = self.res_l1_p(x_p)
        x_e = F.relu(x_e)
        x_p = F.relu(x_p)

        x_e = self.res_l2_e(x_e)
        x_p = self.res_l2_p(x_p)

        out_emb_e = rearrange(x_e, "(b n) d -> b n d", b=batch_size)
        out_emb_p = rearrange(x_p, "(b n) d -> b n d", b=batch_size)
        return out_emb_e, out_emb_p

    def forward(
        self,
        images,
        labels_e = None,
        labels_p = None,
        matrix=None,
        sample_index_e=None,
        sample_index_p=None,
        is_train=True,
        no_cl=False,
        exclude_class=False,
    ):

        # labels batch,51,75 binary_label batch,75 sample_index_e batch,index
        B = images.shape[0]
        device = images.device
        """ Visual Backbone """
        x_e, x_p = self.image_encoder(images)  # batch_size,patch_num,dim
        


        features_e = x_e.transpose(0, 1)  # patch_num b dim
        features_p = x_p.transpose(0, 1)  # patch_num b dim
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        query_embed_e = self.disease_embedding_layer(self.disease_book)
        query_embed_p = self.disease_embedding_layer(self.ana_book)
        query_embed_e = query_embed_e.unsqueeze(1).repeat(1, B, 1)
        query_embed_p = query_embed_p.unsqueeze(1).repeat(1, B, 1)
        features_e, ws_e = self.decoder_e(
            query_embed_e,
            features_e,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )
        features_p, ws_p = self.decoder_p(
            query_embed_p,
            features_p,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )
        if self.use_pe_cl:

            pe_logits = torch.bmm(features_e.transpose(0, 1), features_p.transpose(0, 1).transpose(1, 2)).transpose(1, 2) # B, 51, 75
            matrix_zero = matrix
            matrix_zero[matrix_zero < 1] = 0
            pe_logits = pe_logits.reshape(pe_logits.shape[0]*pe_logits.shape[1]*pe_logits.shape[2], -1)
            matrix_zero = matrix_zero.reshape(matrix_zero.shape[0]*matrix_zero.shape[1]*matrix_zero.shape[2], -1)
            # pe_logits = F.normalize(pe_logits)
            loss_pe = F.binary_cross_entropy_with_logits(pe_logits.float(), matrix_zero.float())
        else:
            loss_pe = torch.tensor(0)

        out_e = self.dropout_feas_e(features_e)
        out_p = self.dropout_feas_p(features_p)
        
        # out = self.dropout_feas(features)
        if is_train == True and no_cl == False:

            # get anatomy query
            anatomy_query = torch.zeros(
                [
                    sample_index_e.shape[0],
                    sample_index_e.shape[1],
                    sample_index_e.shape[2],
                    self.ana_book.shape[-1],
                ]
            ).to(
                device
            )  # [128, 75, 8, 768]
            entity_query = torch.zeros(
                [
                    sample_index_p.shape[0],
                    sample_index_p.shape[1],
                    sample_index_p.shape[2],
                    self.disease_book.shape[-1],
                ]
            ).to(device)

            anatomy_query = self.ana_book[sample_index_e, :] * (
                sample_index_e != -1
            ).int().unsqueeze(-1).repeat(1, 1, 1, 768)  # batch, Q , position_num ,dim [128, 75, 8, 768]
            entity_query = self.disease_book[sample_index_p, :] * (
                sample_index_p != -1
            ).int().unsqueeze(-1).repeat(1, 1, 1, 768)

            # positive_index_e = torch.where(sample_index_e == -1)
            # positive_index_p = torch.where(sample_index_p == -1)

            matrix_zero_e = matrix
            matrix_zero_p = matrix.transpose(1, 2)
            matrix_zero_e[matrix_zero_e < 1] = 0
            matrix_zero_p[matrix_zero_p < 1] = 0
            matrix_zero_e = matrix_zero_e.unsqueeze(3).repeat(
                1, 1, 1, anatomy_query.shape[-1]
            )
            matrix_zero_p = matrix_zero_p.unsqueeze(3).repeat(
                1, 1, 1, entity_query.shape[-1]
            )

            ana_temp = self.ana_book
            dis_temp = self.disease_book
            ana_temp = ana_temp.unsqueeze(0).repeat(anatomy_query.shape[0], 1, 1)
            dis_temp = dis_temp.unsqueeze(0).repeat(entity_query.shape[0], 1, 1)
            ana_temp = ana_temp.unsqueeze(2).repeat(1, 1, anatomy_query.shape[1], 1)
            dis_temp = dis_temp.unsqueeze(2).repeat(1, 1, entity_query.shape[1], 1)

            posi_matrix_e = (matrix_zero_e * ana_temp).transpose(1, 2)
            posi_matrix_p = (matrix_zero_p * dis_temp).transpose(1, 2)

            for i in range(anatomy_query.shape[0]):
                for j in range(anatomy_query.shape[1]):
                    if (posi_matrix_e[i, j] != 0).sum() > 0:
                        num_posi = (
                            torch.nonzero(posi_matrix_e[i, j], as_tuple=True)[0]
                            .unique()
                            .shape[0]
                        )
                        assert anatomy_query[i, j, 0, :].sum() == 0
                        anatomy_query[i, j, 0, :] = (
                            posi_matrix_e[i, j, :, :].sum(dim=0) / num_posi
                        )
            
            for i in range(entity_query.shape[0]):
                for j in range(entity_query.shape[1]):
                    if (posi_matrix_p[i, j] != 0).sum() > 0:
                        num_posi = (
                            torch.nonzero(posi_matrix_p[i, j], as_tuple=True)[0]
                            .unique()
                            .shape[0]
                        )
                        assert entity_query[i, j, 0, :].sum() == 0
                        entity_query[i, j, 0, :] = (
                            posi_matrix_p[i, j, :, :].sum(dim=0) / num_posi
                        )
            # Got anatomy query

            # [Q,B,A]
            ll_e = out_e.transpose(0, 1)  # B Q A
            ll_p = out_p.transpose(0, 1)  # B Q A

            Q_e = ll_e.shape[1]
            Q_p = ll_p.shape[1]

            ll_e = ll_e.reshape(ll_e.shape[0] * ll_e.shape[1], -1)
            ll_p = ll_p.reshape(ll_p.shape[0] * ll_p.shape[1], -1)

            ll_e = self.cl_fc_e(ll_e)
            ll_p = self.cl_fc_p(ll_p)

            ll_e = ll_e.unsqueeze(dim=-1)
            ll_p = ll_p.unsqueeze(dim=-1)

            anatomy_query = anatomy_query.reshape(B * Q_e, 8, 768)
            entity_query = entity_query.reshape(B * Q_p, 8, 768)

            ll_e = torch.bmm(anatomy_query, ll_e).squeeze()  # B Q position_num
            ll_p = torch.bmm(entity_query, ll_p).squeeze()  # B Q position_num

            cl_labels_e = torch.zeros((ll_e.shape[0])).to(device)
            cl_labels_p = torch.zeros((ll_p.shape[0])).to(device)

            if exclude_class == True:
                cl_labels_e = cl_labels_e.reshape(B, Q_e)
                cl_labels_p = cl_labels_p.reshape(B, Q_p)

                cl_labels_e = cl_labels_e[:, self.keep_class_dim_e]
                cl_labels_p = cl_labels_p[:, self.keep_class_dim_e]

                cl_labels_e = cl_labels_e.reshape(-1)
                cl_labels_p = cl_labels_p.reshape(-1)

                ll_e = ll_e.reshape(B, Q_e, -1)
                ll_p = ll_p.reshape(B, Q_p, -1)

                ll_e = ll_e[:, self.keep_class_dim_e, :]
                ll_e = ll_e.reshape(B * (len(self.keep_class_dim_e)), -1)
                ll_p = ll_p.reshape(B * Q_p, -1)
        
        if self.use_pe_cl:
            pe_logits_ = torch.bmm


        x_e = self.classifier_e(out_e).transpose(0, 1)  # []
        x_p = self.classifier_p(out_p).transpose(0, 1)  # B query Atributes

        if exclude_class == True:
            labels_e = labels_e[:, self.keep_class_dim_e]
            x_e = x_e[:, self.keep_class_dim_e, :]

        labels_e = labels_e.reshape(-1, 1)
        labels_p = labels_p.reshape(-1, 1)
        logits_e = x_e.reshape(-1, x_e.shape[-1])
        logits_p = x_p.reshape(-1, x_p.shape[-1])
        Mask_e = ((labels_e != -1) & (labels_e != 2)).squeeze()
        Mask_p = ((labels_p != -1) & (labels_p != 2)).squeeze()

        cl_mask_e = (labels_e == 1).squeeze()
        cl_mask_p = (labels_p == 1).squeeze()
        if is_train == True:
            labels_e = labels_e[Mask_e].long()
            labels_p = labels_p[Mask_p].long()
            logits_e = logits_e[Mask_e]
            logits_p = logits_p[Mask_p]
            loss_ce_e = F.cross_entropy(logits_e, labels_e[:, 0])
            loss_ce_p = F.cross_entropy(logits_p, labels_p[:, 0])
            if no_cl == False:
                cl_labels_e = cl_labels_e[cl_mask_e].long()
                cl_labels_p = cl_labels_p[cl_mask_p].long()
                ll_e = ll_e[cl_mask_e]
                ll_p = ll_p[cl_mask_p]
                loss_cl_e = F.cross_entropy(ll_e, cl_labels_e)
                loss_cl_p = F.cross_entropy(ll_p, cl_labels_p)
                loss_ce = loss_ce_e + loss_ce_p
                loss_cl = loss_cl_e + loss_cl_p
                loss = loss_ce + loss_cl + loss_pe
            else:
                loss_cl = torch.tensor(0)
                loss = loss_ce_e + loss_ce_p + loss_pe
        else:
            loss = 0
        if is_train == True:
            return loss, loss_ce_e, loss_cl_e, loss_ce_p, loss_cl_p, loss_pe
        else:
            return loss, x_e, ws_e, x_p, ws_p

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
