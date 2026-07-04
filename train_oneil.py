
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if Path.cwd() != BASE_DIR:
    print(f"Changing working directory to: {BASE_DIR}")
    os.chdir(BASE_DIR)

CHECKPOINT_DIR = BASE_DIR / "checkpoint"
RESULTS_DIR = BASE_DIR / "fold_results"
LOG_DIR = BASE_DIR / "log"
for output_dir in (CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

print(f"Before setting: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"Command line args: {sys.argv}")

gpu_id = "3"
if "--gpu_id" in sys.argv:
    try:
        idx = sys.argv.index("--gpu_id")
        if idx + 1 < len(sys.argv):
            gpu_id = sys.argv[idx + 1]
    except ValueError:
        pass

print(f"Setting CUDA_VISIBLE_DEVICES to: {gpu_id}")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_oneil import SynergyEncoderDataset
from sklearn.model_selection import KFold, train_test_split

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    from torch.optim.lr_scheduler import LinearLR as get_linear_schedule_with_warmup
from torch.optim import AdamW
from early_stopping import EarlyStopping
from log_utils import setup_logging
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
# from imblearn.metrics import sensitivity_score,specificity_score
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, recall_score, precision_score
import torch.nn.functional as F

from simple_table import PrettyTable
from datetime import datetime
import os
# Import the TMI-Syn model.
from model import TMISyn
import json

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import ClassVar, Sequence, Mapping
from collections import defaultdict
import random

matplotlib.use('Agg')


def set_random_seed(seed=42):
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)

    print(f"Random seed set to: {seed}")


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_oneil_data_file(data_file):
    smiles_1 = []
    smiles_2 = []
    Y = []
    context = []

    with open(data_file, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines[1:]:
            row = line.rstrip().split(',')
            # print(row)
            # 3385,11960529,A2058,1
            smiles_1.append(row[0])
            smiles_2.append(row[1])
            context.append((row[2]))
            Y.append(int(row[4])) # ONeil dataset label is at index 4

    return smiles_2, smiles_1, context, Y


def define_dataloader(train_index, test_index, smiles_1, smiles_2, context, Y, maxCompoundLen, batch_size, seed=42):
    test_index, valid_index = train_test_split(test_index, test_size=0.5, random_state=seed)
    train_drug_2_cv = np.array(smiles_2)[train_index]
    train_drug_1_cv = np.array(smiles_1)[train_index]
    train_context_cv = np.array(context)[train_index]
    train_Y_cv = np.array(Y)[train_index]

    test_drug_2_cv = np.array(smiles_2)[test_index]
    test_drug_1_cv = np.array(smiles_1)[test_index]
    test_context_cv = np.array(context)[test_index]
    test_Y_cv = np.array(Y)[test_index]

    valid_drug_2_cv = np.array(smiles_2)[valid_index]
    valid_drug_1_cv = np.array(smiles_1)[valid_index]
    valid_context_cv = np.array(context)[valid_index]
    valid_Y_cv = np.array(Y)[valid_index]

    train_dataset = SynergyEncoderDataset(train_drug_1_cv,
                                          train_drug_2_cv,
                                          train_Y_cv,
                                          train_context_cv,
                                          maxCompoundLen, device=device)
    
    valid_dataset = SynergyEncoderDataset(valid_drug_1_cv,
                                          valid_drug_2_cv,
                                          valid_Y_cv,
                                          valid_context_cv,
                                          maxCompoundLen, device=device)

    test_dataset = SynergyEncoderDataset(test_drug_1_cv,
                                         test_drug_2_cv,
                                         test_Y_cv,
                                         test_context_cv,
                                         maxCompoundLen, device=device)

    trainLoader = DataLoader(train_dataset,
                             batch_size=batch_size, shuffle=True)

    validLoader = DataLoader(valid_dataset,
                             batch_size=batch_size, shuffle=True)

    testLoader = DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=False)

    return trainLoader, validLoader, testLoader


def validate_new(valid_loader, model):  # valid_loader
    model.eval()
    preds = torch.Tensor()
    trues = torch.Tensor()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # Unpack based on dataset_oneil.py __getitem__ return (10 items)
            compounds_1d_1, compounds_1d_2, synergyScore, context_features, compounds_2d_1, compounds_2d_2, smiles_1, smiles_2, fp1, fp2 = batch

            compounds_1d_1 = compounds_1d_1.to(device)
            compounds_1d_2 = compounds_1d_2.to(device)
            compounds_2d_1 = compounds_2d_1.to(device)
            compounds_2d_2 = compounds_2d_2.to(device)
            context_features = context_features.to(device)
            synergyScore = synergyScore.to(device)
            smiles_1 = smiles_1.to(device)
            smiles_2 = smiles_2.to(device)
            # Add fingerprint vectors to device
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)

            # Modified for Model Version 42:
            pre_synergy, _, _ = model(
                compounds_1d_1,  # drug1_1d_vectors
                compounds_1d_2,  # drug2_1d_vectors
                context_features, # context
                compounds_2d_1,  # drug1_2d_vectors
                compounds_2d_2,  # drug2_2d_vectors
                smiles_1,        # drug1_smi_vectors
                smiles_2,        # drug2_smi_vectors
                fp1,             # drug1_fp_vectors
                fp2              # drug2_fp_vectors
            )

            pre_synergy = torch.nn.functional.softmax(pre_synergy, dim=1)[:, 1]
            preds = torch.cat((preds, pre_synergy.cpu()), 0)
            trues = torch.cat((trues, synergyScore.view(-1, 1).cpu()), 0)
        y_pred = np.array(preds) > 0.5  # np.where(preds>0.5,1,0)
        accuracy = accuracy_score(trues, y_pred)
        BACC = balanced_accuracy_score(trues, y_pred)
        roc_auc = roc_auc_score(trues, preds)
        ACC = accuracy_score(trues, y_pred)
        F1 = f1_score(trues, y_pred, average='binary')
        Prec = precision_score(trues, y_pred, average='binary')
        Rec = recall_score(trues, y_pred, average='binary')
        kappa = cohen_kappa_score(trues, y_pred)
        mcc = matthews_corrcoef(trues, y_pred)
        ap = average_precision_score(trues, preds)

        return accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap


def validate_new_with_preds(valid_loader, model):  # valid_loader
    """Evaluate the model and return metrics, predictions, and labels."""
    model.eval()
    preds = torch.Tensor()
    trues = torch.Tensor()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # Unpack based on dataset_oneil.py __getitem__ return (10 items)
            compounds_1d_1, compounds_1d_2, synergyScore, context_features, compounds_2d_1, compounds_2d_2, smiles_1, smiles_2, fp1, fp2 = batch

            compounds_1d_1 = compounds_1d_1.to(device)
            compounds_1d_2 = compounds_1d_2.to(device)
            compounds_2d_1 = compounds_2d_1.to(device)
            compounds_2d_2 = compounds_2d_2.to(device)
            context_features = context_features.to(device)
            synergyScore = synergyScore.to(device)
            smiles_1 = smiles_1.to(device)
            smiles_2 = smiles_2.to(device)
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)

            # Modified for Model Version 42
            pre_synergy, _, _ = model(
                compounds_1d_1,
                compounds_1d_2,
                context_features,
                compounds_2d_1,
                compounds_2d_2,
                smiles_1,
                smiles_2,
                fp1,
                fp2
            )

            pre_synergy = torch.nn.functional.softmax(pre_synergy, dim=1)[:, 1]
            preds = torch.cat((preds, pre_synergy.cpu()), 0)
            trues = torch.cat((trues, synergyScore.view(-1, 1).cpu()), 0)
        y_pred = np.array(preds) > 0.5  # np.where(preds>0.5,1,0)
        accuracy = accuracy_score(trues, y_pred)
        BACC = balanced_accuracy_score(trues, y_pred)
        roc_auc = roc_auc_score(trues, preds)
        ACC = accuracy_score(trues, y_pred)
        Precision = precision_score(trues, y_pred)
        Recall = recall_score(trues, y_pred)
        F1 = f1_score(trues, y_pred)
        mcc = matthews_corrcoef(trues, y_pred)
        kappa = cohen_kappa_score(trues, y_pred)
        ap = average_precision_score(trues, preds)

        return accuracy, ACC, BACC, Precision, Recall, F1, roc_auc, mcc, kappa, ap, np.array(preds), np.array(
            trues).flatten()


class ConfidenceAwareFocalLoss(nn.Module):
    """
    Confidence-Aware Focal Loss
    Combines Focal Loss with Confidence Budget regularization.
    
    L = FocalLoss(Interpolated_Prob, Target) - lambda * log(Confidence)
    where Interpolated_Prob = Confidence * Prob_Model + (1 - Confidence) * 0.5
    """
    def __init__(self, alpha=1.0, gamma=2.0, lambda_c=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_c = lambda_c
        self.reduction = reduction

    def forward(self, logits, targets, confidence):
        # logits: (B, 2)
        # targets: (B,)
        # confidence: (B, 1)
        
        probs = F.softmax(logits, dim=1)
        confidence = confidence.view(-1, 1)
        
        # Interpolate probabilities
        # If confidence is 1, use model probability. If 0, use 0.5 (random guess).
        probs_new = confidence * probs + (1 - confidence) * 0.5
        
        # Get probability of true class
        # targets: (B,) -> (B, 1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=2).float()
        
        # pt: probability of the correct class
        pt = (probs_new * targets_one_hot).sum(dim=1) # (B,)
        
        # Focal Loss part
        # loss = - alpha * (1-pt)^gamma * log(pt)
        log_pt = torch.log(pt + 1e-8)
        focal_loss = - self.alpha * (1 - pt) ** self.gamma * log_pt
        
        # Confidence penalty: Minimize -log(c) => Maximize c
        conf_penalty = - torch.log(confidence.squeeze(1) + 1e-8)
        
        total_loss_batch = focal_loss + self.lambda_c * conf_penalty
        
        if self.reduction == 'mean':
            return total_loss_batch.mean()
        elif self.reduction == 'sum':
            return total_loss_batch.sum()
        return total_loss_batch


class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE Loss for Contrastive Learning (SimCLR/SimCSE style).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: (B, D) - features from two views of the same batch
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        batch_size = z1.shape[0]
        
        # Concatenate features: (2B, D)
        features = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix: (2B, 2B)
        # sim[i, j] = z_i . z_j / temperature
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create targets:
        # For z1[i] (index i), positive is z2[i] (index i + B)
        # For z2[i] (index i + B), positive is z1[i] (index i)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z1.device)
        
        # Mask out self-similarity (diagonal)
        # We need to make sure the diagonal entries are not selected as negative samples (though exp(sim) would be large)
        # Standard CrossEntropy implementation allows passing logits directly.
        # But we need to exclude the diagonal (self-similarity) from the denominator in softmax?
        # In standard SimCLR, self-similarity is removed.
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix.masked_fill_(mask, -9e15) # Very small number to make exp() close to 0
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


def train(train_loader, model, epoch, optimizer, device, scheduler, print_freq=200, use_focal=True, lambda_c=0.1, lambda_cl=0.1):
    model.train()
    if use_focal:
        loss_fn = ConfidenceAwareFocalLoss(alpha=1.0, gamma=2.0, lambda_c=lambda_c)
    else:
        loss_fn = ConfidenceAwareFocalLoss(alpha=1.0, gamma=0.0, lambda_c=lambda_c)
        
    # Contrastive Loss function
    cl_loss_fn = InfoNCELoss(temperature=0.07)
        
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # Unpack based on dataset_oneil.py __getitem__ return (10 items)
        compounds_1d_1, compounds_1d_2, synergyScore, context_features, compounds_2d_1, compounds_2d_2, smiles_1, smiles_2, fp1, fp2 = batch

        compounds_1d_1 = compounds_1d_1.to(device)
        compounds_1d_2 = compounds_1d_2.to(device)
        compounds_2d_1 = compounds_2d_1.to(device)
        compounds_2d_2 = compounds_2d_2.to(device)
        context_features = context_features.to(device)
        synergyScore = synergyScore.to(device)
        smiles_1 = smiles_1.to(device)
        smiles_2 = smiles_2.to(device)
        fp1 = fp1.to(device)
        fp2 = fp2.to(device)

        # Forward pass 1 (TMI-Syn)
        pre_synergy, confidence1, proj1 = model(
            compounds_1d_1, compounds_1d_2, context_features,
            compounds_2d_1, compounds_2d_2, smiles_1, smiles_2,
            fp1, fp2
        )

        # Forward pass 2 (for R-Drop & Contrastive Learning)
        pre_synergy2, confidence2, proj2 = model(
            compounds_1d_1, compounds_1d_2, context_features,
            compounds_2d_1, compounds_2d_2, smiles_1, smiles_2,
            fp1, fp2
        )

        # Calculate accuracy (based on raw logits of first pass)
        probs = torch.nn.functional.softmax(pre_synergy, dim=1)[:, 1]
        preds = (probs > 0.5).float()
        targets = synergyScore.squeeze(1)
        acc = (preds == targets).float().mean()
        accuracies.update(acc.item(), len(compounds_1d_1))

        # Calculate Losses
        # 1. Main Task Loss (Confidence Aware)
        loss1 = loss_fn(pre_synergy, targets, confidence1)
        loss2 = loss_fn(pre_synergy2, targets, confidence2)
        ce_loss = 0.5 * (loss1 + loss2)
        
        # 2. Contrastive Loss (InfoNCE)
        cl_loss = cl_loss_fn(proj1, proj2)
        
        # 3. KL Divergence (Consistency Loss) - on logits
        # We can keep this or remove it. Since we have CL, we might reduce weight or keep it.
        # Let's keep it as it stabilizes predictions.
        kl_loss = compute_kl_loss(pre_synergy, pre_synergy2)
        
        alpha_kl = 5
        # Total Loss = Task + alpha * KL + lambda * Contrastive
        loss = ce_loss + alpha_kl * kl_loss + lambda_cl * cl_loss

        losses.update(loss.item(), len(compounds_1d_1))
        loss.backward()
        if np.isnan(loss.item()):
            raise Exception("Training model diverges.")
        optimizer.step()
        scheduler.step()

        if i % print_freq == 0:
            log_str = 'TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})\t Acc:{acc.val:.4f} ({acc.avg:.4f})'.format(epoch=epoch,
                                                                                                           iter=i,
                                                                                                           loss=losses,
                                                                                                           acc=accuracies)
            print(log_str)
            
    return losses.avg, accuracies.avg


def run_expriments(device, model_type='TMI-Syn', seed=42, use_focal=True, checkpoint_name='checkpoint',dataset_path='ONeil/ONeil.csv',lr = 1e-3,batch_size = 256, weight_decay=1e-4, dropout=0.2, lambda_c=0.1, lambda_cl=0.1):
    set_random_seed(seed)

    all_loss = np.zeros((5, 1))
    all_accuracy = np.zeros((5, 1))
    all_report = np.zeros((5, 1))
    all_ACC = np.zeros((5, 1))
    all_BACC = np.zeros((5, 1))
    all_Prec = np.zeros((5, 1))
    all_Rec = np.zeros((5, 1))
    all_F1 = np.zeros((5, 1))
    all_roc_auc = np.zeros((5, 1))
    all_mcc = np.zeros((5, 1))
    all_kappa = np.zeros((5, 1))
    all_ap = np.zeros((5, 1))
    n_epochs = 200
    
    # ONeil dataset logic
    smiles_1, smiles_2, context, Y = read_oneil_data_file(dataset_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for split, (train_index, test_index) in enumerate(kf.split(Y)):
        # maxCompoundLen default to 128
        trainLoader, validLoader, testLoader = define_dataloader(train_index, test_index, smiles_1, smiles_2, context,
                                                                 Y, 128, batch_size, seed)
        
        # Initialize TMI-Syn.
        model = TMISyn(dropout_rate=dropout)
        
        print(f"Using TMI-Syn with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Hyperparameters: LR={lr}, WD={weight_decay}, Dropout={dropout}, Lambda_c={lambda_c}, Lambda_cl={lambda_cl}")
        current_checkpoint_name = str(CHECKPOINT_DIR / f"{checkpoint_name}_fold{split}.pt")
        model.to(device)
        stopper = EarlyStopping(mode='higher', filename=current_checkpoint_name, patience=60)
        model = model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=n_epochs,
                                                        steps_per_epoch=len(trainLoader))
        for epochind in range(n_epochs):
            train_loss, train_acc = train(trainLoader, model, epochind, optimizer, device, scheduler, print_freq=200, use_focal=use_focal, lambda_c=lambda_c, lambda_cl=lambda_cl)
            
            print(f"\n[Epoch {epochind} Summary] Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.4f}")

            # 1. Validate on Validation Set
            accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = validate_new(validLoader, model)
            e_tables = PrettyTable(['epoch (Val)', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap'])
            e_tables.float_format = '.3'
            row = [epochind, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
            e_tables.add_row(row)
            print(f"\nFold {split} - Epoch {epochind} - Validation Results:")
            print(e_tables)
            
            # 2. Validate on Test Set
            t_accuracy, t_ACC, t_BACC, t_Prec, t_Rec, t_F1, t_roc_auc, t_mcc, t_kappa, t_ap = validate_new(testLoader, model)
            t_tables = PrettyTable(['epoch (Test)', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap'])
            t_tables.float_format = '.3'
            t_row = [epochind, t_ACC, t_BACC, t_Prec, t_Rec, t_F1, t_roc_auc, t_mcc, t_kappa, t_ap]
            t_tables.add_row(t_row)
            print(f"Fold {split} - Epoch {epochind} - Test Set Results:")
            print(t_tables)
            print("-" * 60)

            # Early Stopping Check (based on Validation ACC)
            early_stop = stopper.step(ACC, model)
            if early_stop:
                break
        
        stopper.load_checkpoint(model)
        accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap, test_preds, test_labels = validate_new_with_preds(
            testLoader, model)

        e_tables = PrettyTable(['Final Test', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap'])
        e_tables.float_format = '.3'
        row = [epochind, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
        e_tables.add_row(row)
        print("\nFinal Test Results (Best Validation Model):")
        print(e_tables)

        results_dict = {
            'accuracy': accuracy,
            'ACC': ACC,
            'BACC': BACC,
            'Prec': Prec,
            'Rec': Rec,
            'F1': F1,
            'roc_auc': roc_auc,
            'mcc': mcc,
            'kappa': kappa,
            'ap': ap,
            'test_preds': test_preds,
            'test_labels': test_labels
        }
        # save_results_to_csv(results_dict, model_type, split, output_dir="fold_results")

        all_accuracy[split] = accuracy
        all_ACC[split] = ACC
        all_BACC[split] = BACC
        all_Prec[split] = Prec
        all_Rec[split] = Rec
        all_F1[split] = F1
        all_roc_auc[split] = roc_auc
        all_mcc[split] = mcc
        all_kappa[split] = kappa
        all_ap[split] = ap

    print('*=' * 20)
    print('accuracy:  {0:6f}({1:6f})'.format(np.mean(all_accuracy), np.std(all_accuracy)))
    print('ACC:  {0:6f}({1:6f})'.format(np.mean(all_ACC), np.std(all_ACC)))
    print('BACC:  {0:6f}({1:6f})'.format(np.mean(all_BACC), np.std(all_BACC)))
    print('Prec:  {0:6f}({1:6f})'.format(np.mean(all_Prec), np.std(all_Prec)))
    print('Rec:  {0:6f}({1:6f})'.format(np.mean(all_Rec), np.std(all_Rec)))
    print('F1:  {0:6f}({1:6f})'.format(np.mean(all_F1), np.std(all_F1)))
    print('roc_auc:  {0:6f}({1:6f})'.format(np.mean(all_roc_auc), np.std(all_roc_auc)))
    print('mcc:  {0:6f}({1:6f})'.format(np.mean(all_mcc), np.std(all_mcc)))
    print('kappa:  {0:6f}({1:6f})'.format(np.mean(all_kappa), np.std(all_kappa)))
    print('ap:  {0:6f}({1:6f})'.format(np.mean(all_ap), np.std(all_ap)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = RESULTS_DIR / f"kfold_summary_{model_type}_{timestamp}.csv"
    
    # Ensure directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_data = {
        'Model_Type': [model_type] * 10,
        'Metric': ['accuracy', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'roc_auc', 'mcc', 'kappa', 'ap'],
        'Mean': [
            np.mean(all_accuracy), np.mean(all_ACC), np.mean(all_BACC),
            np.mean(all_Prec), np.mean(all_Rec), np.mean(all_F1),
            np.mean(all_roc_auc), np.mean(all_mcc), np.mean(all_kappa), np.mean(all_ap)
        ],
        'Std': [
            np.std(all_accuracy), np.std(all_ACC), np.std(all_BACC),
            np.std(all_Prec), np.std(all_Rec), np.std(all_F1),
            np.std(all_roc_auc), np.std(all_mcc), np.std(all_kappa), np.std(all_ap)
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_filename, index=False)
    print(f"K-Fold summary saved to: {summary_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ONeil drug synergy prediction with TMI-Syn')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='GPU ID to use (only for CUDA)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda_c', type=float, default=0.1, help='Lambda for confidence loss')
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='Lambda for contrastive loss')
    
    # 42 100 1
    args = parser.parse_args()
    log_path = setup_logging(LOG_DIR / f"train_oneil_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Log file: {log_path}")

    checkpoint_name = f'TMI-Syn_ONeil_seed{args.seed}_wd{args.weight_decay}_drop{args.dropout}_lr{args.lr}_lamc{args.lambda_c}_lamcl{args.lambda_cl}'
    dataset_path = 'ONeil/ONeil.csv'
    batch_size = 256
    lr = args.lr
    print(f"\n{'=' * 50}")
    print(f"Running ONeil dataset with TMI-Syn")
    print(f"Device: {device}")
    print(f"Random Seed: {args.seed}")
    print(f"Hyperparameters: LR={lr}, WD={args.weight_decay}, Dropout={args.dropout}, Lambda_c={args.lambda_c}, Lambda_cl={args.lambda_cl}")
    print(f"checkpoint name: {checkpoint_name}")
    print(f"{'=' * 50}\n")

    run_expriments(device, seed=args.seed, use_focal=True, checkpoint_name=checkpoint_name,dataset_path=dataset_path,lr=lr,batch_size=batch_size, weight_decay=args.weight_decay, dropout=args.dropout, lambda_c=args.lambda_c, lambda_cl=args.lambda_cl)
