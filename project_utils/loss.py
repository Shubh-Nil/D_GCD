import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class GradReverse(Function):
    def __init__(self, lambd: int):
        self.lambd = lambd
    
    @staticmethod
    def forward(self, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1)
    
def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd).apply(x)

class Discriminator(nn.Module):
    def __init__(self, output_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features = output_features,
                      out_features = output_features//2),
            nn.LeakyReLU(),
            nn.Linear(in_features = output_features//2,
                      out_features = output_features//4),
            nn.LeakyReLU(),
            nn.Linear(in_features = output_features//4,
                      out_features = output_features//8),
            nn.LeakyReLU(),
            nn.Linear(in_features = output_features//8,
                      out_features = output_features//4),
            nn.LeakyReLU(),
            nn.Linear(in_features = output_features//4,
                      out_features = output_features//2),
            nn.LeakyReLU(),
            nn.Linear(in_features = output_features//2,
                      out_features = output_features),
            nn.LeakyReLU()
        )
        self.lambd = 1

    def set_lambda(self, lambd: int = 1):
        self.lambd = lambd

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            x = grad_reverse(x, self.lambd)
            out = self.network(x)
        else:
            out = self.network(x)
        return out

class AdverserialLoss(nn.Module):
    def __init__(self, num_classes, device, temperature=0.1, alpha=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.y_t = torch.Tensor([1 / self.num_classes for _ in range(num_classes)]).to(device)     # Shape :(num_labelled_classes)
        self.device = device

    def forward(self, source_entropy, target_entropy, labelled_or_not, labels, disc=True):
        y_source = labels[labelled_or_not.bool()].long().to(self.device)
        # unique_labels = y_source.unique()
        # label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
        # # Convert to contiguous labels
        # converted_labels = torch.tensor([label_mapping[label.item()] for label in y_source]).to(self.device)
        source_loss = nn.CrossEntropyLoss()(source_entropy.to(self.device),y_source)
        target_entropy = F.softmax(target_entropy / self.temperature, dim=1).to(self.device)
        y_target = self.y_t.expand(target_entropy.shape[0], -1).to(self.device)
        target_loss = torch.mean(-y_target * torch.log(target_entropy))
        # target_loss = nn.CrossEntropyLoss()(target_entropy / self.temperature, y_target.argmax(dim=1).to(self.device))

        if disc:
            return source_loss + target_loss / self.alpha
        else:
            return source_loss - target_loss / self.alpha

def info_nce_logits(features,device,temperature=0.7,n_views=2):
    b_ = 0.5 * int(features.size(0))                                                                # b_ = 64
    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)                           # len(labels) = 128, [0,...63,0,....63]
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()                                   # labels become 128x128 matrix
    labels = labels.to(device)

    features = F.normalize(features, dim=1)                                                         # 128 x 65536 (after coming out from the projection head)

    similarity_matrix = torch.matmul(features, features.T)                                          # similarity_matrix.shape = 128x128
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)                                  # mask is an Identity matrix of shape 128x128
    labels = labels[~mask].view(labels.shape[0], -1)                                                # mask with diagonal elements = 0      
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)               # similarity matrix with diagonal elements = 0
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)                          # positives.shape = 128x128
                                                                                                    # view i with view j of the same class will have 1, rest will be 0
                                                                                                    # (view i and view i  of the same class = 0 (diagonal = 0))
    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)              # negatives.shape = 128x128
                                                                                                    # opposite to positives, remaining will be 1, view i with view j = 0
    # print(negatives.shape)

    logits = torch.cat([positives, negatives], dim=1)                                               # logits.shape = (128, 256)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)                              # labels.shape = 128

    logits = logits / temperature
    return logits, labels

def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)