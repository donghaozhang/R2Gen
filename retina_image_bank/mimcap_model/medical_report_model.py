import torch
import torch.nn as nn
import numpy as np
from .visual_extractor import VisualExtractor
from .encoder_decoder import EncoderDecoder, EncoderDecoderAug, EncoderDecoderAugv2, EncoderDecoderAugv3
from .encoder_decoder import EncoderDecoderAbv1, EncoderDecoderAugv3Abrm

# Find the location R2GenModelAugv3AbrmDanliDatav2
class R2GenModelAugv3AbrmDanliDatav2(nn.Module):
    def __init__(self, args, tokenizer):
        # print('R2GenModelAug class is being called')
        super(R2GenModelAugv3AbrmDanliDatav2, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoderAugv3Abrm(args, tokenizer)
        # if args.dataset_name == 'iu_xray':
        #     self.forward = self.forward_iu_xray
        # else:
        #     self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output
