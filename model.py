import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from longclip import longclip


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MTPG(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.maf_num=1
        self.model, _ = longclip.load("checkpoints/longclip-L.pt", device=device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.half_input_dim = int(input_dim / 2)
        self.query_text=nn.Linear(input_dim, input_dim)
        self.key_text = nn.Linear(input_dim, input_dim)
        self.value_text = nn.Linear(input_dim, input_dim)

        self.query_image = nn.Linear(input_dim, input_dim)
        self.key_image = nn.Linear(input_dim, input_dim)
        self.value_image = nn.Linear(input_dim, input_dim)

        self.query_image_s = nn.Linear(input_dim, input_dim)
        self.key_image_s= nn.Linear(input_dim, input_dim)
        self.value_image_s= nn.Linear(input_dim, input_dim)

        self.query_text_s = nn.Linear(input_dim, input_dim)
        self.key_text_s = nn.Linear(input_dim, input_dim)
        self.value_text_s = nn.Linear(input_dim, input_dim)

        self.query_image_s_2 = nn.Linear(input_dim, input_dim)
        self.key_image_s_2= nn.Linear(input_dim, input_dim)
        self.value_image_s_2= nn.Linear(input_dim, input_dim)

        self.query_text_2 = nn.Linear(input_dim, input_dim)
        self.key_text_s_2= nn.Linear(input_dim, input_dim)
        self.value_text_s_2 = nn.Linear(input_dim, input_dim)

        self.fc1 = nn.Linear(int(2*input_dim), input_dim)
        self.fc2= nn.Linear(int(2*input_dim), input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)
        self.fc4 = nn.Linear(input_dim, input_dim)
        self.dim =input_dim
        self.avg_pool=nn.AvgPool1d(2,2)
        self.exband1=nn.Linear(int(input_dim/2), input_dim)
        self.exband2 = nn.Linear(int(input_dim/2), input_dim)


        self.ffn = nn.Sequential(
            nn.Linear(input_dim,self.half_input_dim),
            nn.ReLU(),
            nn.Linear(self.half_input_dim, input_dim ),
        )
        self.mlp = nn.Linear(int(input_dim*2), input_dim)
        self.sigmoid=nn.Sigmoid()

    def intra_update(self,image_feature,info_feature):
        quary_text_s= self.query_text_s(info_feature)
        key_text_s = self.key_text_s(info_feature)
        value_text_s = self.value_text_s(info_feature)
        attn_scores_text_to_text = torch.matmul(quary_text_s, key_text_s.transpose(-2, -1))
        self_info_feature = torch.matmul(attn_scores_text_to_text, value_text_s)
        info_feature_avg=self.avg_pool(self_info_feature.unsqueeze(1))
        gate_info_to_image=self.exband1(info_feature_avg.squeeze(1))
        gate_info_to_image = self.sigmoid(gate_info_to_image)

        quary_image_s = self.query_image_s(image_feature)
        key_image_s = self.key_image_s(image_feature)
        value_image_s = self.value_image_s(image_feature)
        attn_scores_image_to_image = torch.matmul(quary_image_s, key_image_s.transpose(-2, -1))
        self_image_feature = torch.matmul(attn_scores_image_to_image, value_image_s)
        image_feature_avg= self.avg_pool(self_image_feature.unsqueeze(1))
        gate_image_to_info=self.exband2(image_feature_avg.squeeze(1))
        gate_image_to_info=self.sigmoid(gate_image_to_info)

        quary_text_new = (1 + gate_image_to_info) * quary_text_s
        key_text_new = (1 + gate_image_to_info) * key_text_s
        quary_image_new = (1 + gate_info_to_image) * quary_image_s
        key_image_new = (1 + gate_info_to_image) * key_image_s

        attn_scores_text_to_text_2= torch.matmul(quary_text_new, key_text_new.transpose(-2, -1))
        info_feature_update= torch.matmul(attn_scores_text_to_text_2, value_text_s)

        attn_scores_image_to_image_2= torch.matmul(quary_image_new, key_image_new.transpose(-2, -1))
        image_feature_update= torch.matmul(attn_scores_image_to_image_2, value_image_s)
        image_feature=self.fc3(image_feature_update)
        info_feature = self.fc4(info_feature_update)

        return image_feature,info_feature

    def inter_update(self,image_feature,info_feature):
        quary_text = self.query_text(info_feature)
        key_image = self.key_image(image_feature)
        value_image = self.value_image(image_feature)
        attn_scores_text_to_image = torch.matmul(quary_text, key_image.transpose(-2, -1))
        cross_info_feature = torch.matmul(attn_scores_text_to_image, value_image)
        image_enhanced_info_feature = self.fc1(torch.cat((info_feature,cross_info_feature),1))

        quary_image = self.query_text(image_feature)
        key_text = self.key_image(info_feature)
        value_text = self.value_image(info_feature)
        attn_scores_image_to_text = torch.matmul(quary_image,
                                                 key_text.transpose(-2, -1))
        cross_image_feature = torch.matmul(attn_scores_image_to_text,value_text)
        info_enhanced_image_feature = self.fc2(torch.cat((image_feature,cross_image_feature),1))
        return info_enhanced_image_feature,image_enhanced_info_feature


    def forward(self, x, info, aes_prompt):
        image_feature = self.model.encode_image(x)
        info_feature = self.model.encode_text(info)
        aes_prompt_feature = self.model.encode_text(aes_prompt)

        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
        image_feature = image_feature.float()
        info_feature = info_feature / info_feature.norm(dim=1, keepdim=True)
        info_feature = info_feature.float()
        aes_prompt_feature = aes_prompt_feature / aes_prompt_feature.norm(dim=1, keepdim=True)
        aes_prompt_feature = aes_prompt_feature.float()
        image_feature_input=image_feature
        info_feature_input=info_feature


        for i in range(self.maf_num):
            image_feature_o=image_feature_input
            info_enhanced_image_feature, image_enhanced_info_feature=self.inter_update(image_feature_input,info_feature)
            image_feature_input, info_feature_input_= self.intra_update(info_enhanced_image_feature,image_enhanced_info_feature)
            info_enhanced_image_feature_, image_enhanced_info_feature_= self.inter_update(image_feature,
                                                                                         info_feature)
            image_feature_input_, info_feature_input = self.intra_update(info_enhanced_image_feature_,
                                                                        image_enhanced_info_feature_)


        residual1 = self.mlp(torch.cat((info_feature_input, image_feature_input), 1))
        z=torch.sigmoid(residual1)
        image_feature = (1 - z) * image_feature + z * residual1

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_feature @ aes_prompt_feature.t()
        logits_per_image = F.softmax(logits_per_image, dim=1)

        return logits_per_image

