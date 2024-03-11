import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from einops import rearrange

class ModelRes_ft(nn.Module):
    def __init__(
        self, res_base_model, out_size, imagenet_pretrain=False, linear_probe=False, use_base=True
    ):
        super(ModelRes_ft, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=imagenet_pretrain),
            "resnet50": models.resnet50(pretrained=imagenet_pretrain),
        }
        resnet = self._get_res_basemodel(res_base_model)
        self.use_base = use_base    

        if not self.use_base:
            num_ftrs = int(resnet.fc.in_features/2)
            self.res_features = nn.Sequential(*list(resnet.children())[:-3])
            self.res_l1_p = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2_p = nn.Linear(num_ftrs, 256)
            self.res_l1_e = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2_e = nn.Linear(num_ftrs, 256)

            self.mask_generator = nn.Linear(num_ftrs, num_ftrs)
            self.back = nn.Linear(256, num_ftrs)
            self.last_res = nn.Sequential(*list(resnet.children())[-3:-1])
        else:
            self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        self.res_out = nn.Linear(int(resnet.fc.in_features), out_size)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"
            )
    
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
        x_e = mask * x
        # x_p = (1 - mask) * x
        x_e = self.res_l1_e(x_e)
        # x_p = self.res_l1_p(x_p)
        x_e = F.relu(x_e)
        # x_p = F.relu(x_p)

        x_e = self.res_l2_e(x_e)
        # x_p = self.res_l2_p(x_p)

        out_emb_e = rearrange(x_e, "(b n) d -> b n d", b=batch_size)
        out_emb_e = self.back(out_emb_e)
        out_emb_e = rearrange(out_emb_e, "b (n1 n2) d -> b d n1 n2", n1=14, n2=14)
        # out_emb_e = rearrange(out_emb_e, "b d n -> (b n) d")
        out_emb_e = self.last_res(out_emb_e)
        out_emb_e = out_emb_e.squeeze()
        # out_emb_e = self.res_out(out_emb_e)
        # out_emb_p = rearrange(x_p, "(b n) d -> b n d", b=batch_size)
        return out_emb_e

    def forward(self, img, linear_probe=False):
        if self.use_base:
            x = self.res_features(img)
        else:
            x = self.image_encoder(img)

        x = x.squeeze()
        if linear_probe:
            return x
        else:
            x = self.res_out(x)
            return x
