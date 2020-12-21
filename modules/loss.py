import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        # print('the size of input', input.size(), 'the size of target', target.size(), 'the size of mask', mask.size())
        # print('the size of target before indexing', target.size())
        target = target[:, :input.size(1)]
        # print('the size of target after indexing', target.size())
        mask = mask[:, :input.size(1)]
        # print('the size of mask after indexing', mask.size())
        # print('the size of input before gathering', input.size())
        # print('target.long().unsqueeze(2)', target.long().unsqueeze(2).size())
        # print('-input.gather(2, target.long().unsqueeze(2))', input.gather(2, target.long().unsqueeze(2)).size())
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        # print('the current size of output', output.size())
        output = torch.sum(output) / torch.sum(mask)
        # print('final output', output.size(), output)
        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    # print('loss operation', loss.size(), loss)
    return loss