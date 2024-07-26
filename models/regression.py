import torch
import torch.nn as nn
import torchvision.models as models
from .blocks import SEBlock

class ModifiedResNet101(nn.Module):
    def __init__(self):
        super(ModifiedResNet101, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet101.children())[:-2])
        for name, module in self.features.named_children():
            if 'layer' in name:
                module.add_module("se_block", SEBlock(module[-1].conv2.out_channels))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.regressor = nn.Linear(resnet101.fc.in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.regressor(x)
        return x

def CoxLoss(survtime, censor, hazard_pred, device, model, lambda_reg=0.5):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)

    l2_reg = None
    for param in model.parameters():
        if l2_reg is None:
            l2_reg = 0.5 * param.norm(2)
        else:
            l2_reg = l2_reg + 0.5 * param.norm(2)

    total_loss = loss_cox + lambda_reg * l2_reg
    return total_loss

def accuracy_cox(hazardsdata, labels):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)

def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))
