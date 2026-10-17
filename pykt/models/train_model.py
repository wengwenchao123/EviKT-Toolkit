import os
import torch
from torch.nn.functional import one_hot, binary_cross_entropy
import numpy as np
from .evaluate_model import evaluate, evaluate_edl_stats
from .edl import dirichlet_from_logits, edl_mse_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cal_loss(model, ys, r, rshft, sm, preloss=None, epoch_num=0):
    if preloss is None:
        preloss = []
    model_name = model.model_name
    if model_name in ['simplekt', 'keenkt']:
        if getattr(model, 'use_edl', False):
            logits = ys[1]
            if logits.ndim != 3 or logits.size(-1) != model.num_classes:
                raise ValueError(f'{model_name.upper()} EDL logits must have shape [B,T,{model.num_classes}], got {tuple(logits.shape)}.')
            if sm.dtype != torch.bool:
                raise TypeError(f'{model_name.upper()} EDL smask must be bool, got {sm.dtype}.')
            if logits.shape[:2] != sm.shape:
                raise ValueError(f'{model_name.upper()} EDL logits and smask are not aligned: logits={logits.shape[:2]}, smask={sm.shape}.')
            if rshft.shape != sm.shape:
                raise ValueError(f'{model_name.upper()} EDL targets and smask are not aligned: targets={rshft.shape}, smask={sm.shape}.')
            valid_logits = logits[sm]
            valid_targets = torch.masked_select(rshft, sm).long()
            if valid_logits.size(0) != valid_targets.size(0):
                raise RuntimeError(f'{model_name.upper()} EDL logits and targets have different sample counts.')
            if valid_targets.numel() == 0:
                raise ValueError(f'No valid {model_name.upper()} EDL samples remain after smask.')
            if not torch.all((valid_targets == 0) | (valid_targets == 1)):
                invalid_targets = torch.unique(valid_targets[(valid_targets != 0) & (valid_targets != 1)])
                raise ValueError(f'{model_name.upper()} EDL targets must contain only 0 or 1, but found {invalid_targets.tolist()}.')
            target_one_hot = one_hot(valid_targets, num_classes=model.num_classes).to(device=valid_logits.device, dtype=valid_logits.dtype)
            edl_loss, components = edl_mse_loss(logits=valid_logits, target_one_hot=target_one_hot, epoch_num=epoch_num, annealing_step=model.edl_annealing_step, num_classes=model.num_classes, return_components=True, loss_type=getattr(model, 'edl_loss_type', 'mse'))
            loss = edl_loss
            auxiliary_loss = valid_logits.new_zeros(())
            if model_name == 'keenkt':
                auxiliary_loss = ys[2]
                if not torch.is_tensor(auxiliary_loss):
                    auxiliary_loss = valid_logits.new_tensor(float(auxiliary_loss))
                if auxiliary_loss.ndim != 0:
                    auxiliary_loss = auxiliary_loss.mean()
                loss = loss + auxiliary_loss
            model._last_edl_stats = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in components.items()}
            if model_name == 'keenkt':
                model._last_edl_stats['cl_loss'] = float(auxiliary_loss.detach().cpu())
                model._last_edl_stats['total_with_cl'] = float(loss.detach().cpu())
        else:
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
            loss1 = binary_cross_entropy(y.double(), t.double())
            if model_name == 'keenkt':
                auxiliary_loss = ys[1]
                if not torch.is_tensor(auxiliary_loss):
                    auxiliary_loss = y.new_tensor(float(auxiliary_loss))
                if auxiliary_loss.ndim != 0:
                    auxiliary_loss = auxiliary_loss.mean()
                loss = loss1 + auxiliary_loss
            elif model.emb_type.find('predcurc') != -1:
                if model.emb_type.find('his') != -1:
                    loss = model.l1 * loss1 + model.l2 * ys[1] + model.l3 * ys[2]
                else:
                    loss = model.l1 * loss1 + model.l2 * ys[1]
            elif model.emb_type.find('predhis') != -1:
                loss = model.l1 * loss1 + model.l2 * ys[1]
            else:
                loss = loss1
    elif model_name == 'atdkt':
        if getattr(model, 'use_edl', False):
            logits = ys[1]
            if logits.ndim != 3 or logits.size(-1) != model.num_classes:
                raise ValueError(f'ATDKT EDL selected logits must have shape [B,T,{model.num_classes}], got {tuple(logits.shape)}.')
            if sm.dtype != torch.bool:
                raise TypeError(f'ATDKT EDL smask must be bool, got {sm.dtype}.')
            if logits.shape[:2] != sm.shape:
                raise ValueError(f'ATDKT EDL logits and smask are not aligned: logits={logits.shape[:2]}, smask={sm.shape}.')
            if rshft.shape != sm.shape:
                raise ValueError(f'ATDKT EDL targets and smask are not aligned: targets={rshft.shape}, smask={sm.shape}.')
            valid_logits = logits[sm]
            valid_targets = torch.masked_select(rshft, sm).long()
            if valid_logits.size(0) != valid_targets.size(0):
                raise RuntimeError('ATDKT EDL logits and targets have different sample counts.')
            if valid_targets.numel() == 0:
                raise ValueError('No valid ATDKT EDL samples remain after smask.')
            if not torch.all((valid_targets == 0) | (valid_targets == 1)):
                raise ValueError('ATDKT EDL targets must contain only 0 or 1.')
            target_one_hot = one_hot(valid_targets, num_classes=model.num_classes).to(device=valid_logits.device, dtype=valid_logits.dtype)
            edl_loss, components = edl_mse_loss(logits=valid_logits, target_one_hot=target_one_hot, epoch_num=epoch_num, annealing_step=model.edl_annealing_step, num_classes=model.num_classes, return_components=True, loss_type=getattr(model, 'edl_loss_type', 'mse'))
            if model.emb_type.find('predcurc') != -1:
                if model.emb_type.find('his') != -1:
                    loss = model.l1 * edl_loss + model.l2 * ys[2] + model.l3 * ys[3]
                else:
                    loss = model.l1 * edl_loss + model.l2 * ys[2]
            elif model.emb_type.find('predhis') != -1:
                loss = model.l1 * edl_loss + model.l2 * ys[2]
            else:
                loss = edl_loss
            model._last_edl_stats = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in components.items()}
            model._last_edl_stats['cl_loss'] = 0.0
            model._last_edl_stats['total_with_cl'] = float(loss.detach().cpu())
            if not getattr(model, '_edl_training_verified', False):
                with torch.no_grad():
                    _, alpha, probabilities, uncertainty = dirichlet_from_logits(valid_logits, num_classes=model.num_classes)
                    probability_sums = probabilities.sum(dim=-1)
                    assert torch.isfinite(probabilities).all()
                    assert torch.all(alpha >= 1.0)
                    assert torch.all(uncertainty > 0.0)
                    assert torch.all(uncertainty <= 1.0)
                    assert torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=1e-06, rtol=1e-06)
                print('\n[ATDKT+EDL complete training-path verification]')
                print(f'valid logits shape : {tuple(valid_logits.shape)}')
                print(f'probability range  : [{probabilities.min().item():.8f}, {probabilities.max().item():.8f}]')
                print(f'probability sum    : [{probability_sums.min().item():.8f}, {probability_sums.max().item():.8f}]')
                print(f'uncertainty range  : [{uncertainty.min().item():.8f}, {uncertainty.max().item():.8f}]')
                print(f'PASS: ATDKT+EDL loss {edl_loss.item():.8f}\n')
                model._edl_training_verified = True
        else:
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
            loss1 = binary_cross_entropy(y.double(), t.double())
            if model.emb_type.find('predcurc') != -1:
                if model.emb_type.find('his') != -1:
                    loss = model.l1 * loss1 + model.l2 * ys[1] + model.l3 * ys[2]
                else:
                    loss = model.l1 * loss1 + model.l2 * ys[1]
            elif model.emb_type.find('predhis') != -1:
                loss = model.l1 * loss1 + model.l2 * ys[1]
            else:
                loss = loss1
    elif model_name == 'ukt':
        if model.use_edl:
            logits = ys[1]
            if logits.ndim != 3 or logits.size(-1) != model.num_classes:
                raise ValueError('UKT EDL logits should have shape [B, T, {}], got {}.'.format(model.num_classes, tuple(logits.shape)))
            if sm.dtype != torch.bool:
                raise TypeError(f'UKT EDL smask must be a bool tensor, got {sm.dtype}.')
            if logits.shape[:2] != sm.shape:
                raise ValueError(f'UKT EDL logits and smask are not aligned: logits sequence shape={logits.shape[:2]}, smask shape={sm.shape}.')
            if rshft.shape != sm.shape:
                raise ValueError(f'UKT EDL targets and smask are not aligned: target shape={rshft.shape}, smask shape={sm.shape}.')
            valid_logits = logits[sm]
            valid_targets = torch.masked_select(rshft, sm).long()
            if valid_logits.size(0) != valid_targets.size(0):
                raise RuntimeError(f'Masked EDL logits and targets are not aligned: logits count={valid_logits.size(0)}, target count={valid_targets.size(0)}.')
            if valid_targets.numel() == 0:
                raise ValueError('No valid EDL targets remain after smask.')
            if not torch.all((valid_targets == 0) | (valid_targets == 1)):
                invalid_targets = torch.unique(valid_targets[(valid_targets != 0) & (valid_targets != 1)])
                raise ValueError(f'UKT EDL targets must contain only 0 or 1, but found {invalid_targets.tolist()}.')
            target_one_hot = one_hot(valid_targets, num_classes=model.num_classes).to(device=valid_logits.device, dtype=valid_logits.dtype)
            edl_loss, components = edl_mse_loss(logits=valid_logits, target_one_hot=target_one_hot, epoch_num=epoch_num, annealing_step=model.edl_annealing_step, num_classes=model.num_classes, return_components=True, loss_type=getattr(model, 'edl_loss_type', 'mse'))
            if model.use_CL:
                cl_loss = ys[2]
                loss = edl_loss + model.cl_weight * cl_loss
            else:
                cl_loss = None
                loss = edl_loss
            model._last_edl_stats = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in components.items()}
            model._last_edl_stats['cl_loss'] = float(cl_loss.detach().cpu()) if cl_loss is not None else 0.0
            model._last_edl_stats['total_with_cl'] = float(loss.detach().cpu())
            if not getattr(model, '_edl_training_verified', False):
                with torch.no_grad():
                    verification_input = valid_logits
                    verification_input_name = 'valid logits shape'
                    _, alpha, probabilities, uncertainty = dirichlet_from_logits(verification_input, num_classes=model.num_classes)
                    probability_sums = probabilities.sum(dim=-1)
                    assert torch.isfinite(probabilities).all(), 'EDL probabilities contain NaN or Inf.'
                    assert torch.all(probabilities >= 0.0), 'EDL probability is smaller than 0.'
                    assert torch.all(probabilities <= 1.0), 'EDL probability is greater than 1.'
                    assert torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=1e-06, rtol=1e-06), 'P(wrong) + P(correct) is not equal to 1.'
                    assert torch.all(alpha >= 1.0), 'Dirichlet alpha must be greater than or equal to 1.'
                    assert torch.all(uncertainty > 0.0), 'EDL uncertainty must be greater than 0.'
                    assert torch.all(uncertainty <= 1.0), 'EDL uncertainty must not be greater than 1.'
                print(f'\n[{model.model_name}+EDL complete training-path verification]')
                print(f'{verification_input_name} : {tuple(verification_input.shape)}')
                print(f'probability range  : [{probabilities.min().item():.8f}, {probabilities.max().item():.8f}]')
                print(f'probability sum    : [{probability_sums.min().item():.8f}, {probability_sums.max().item():.8f}]')
                print(f'uncertainty range  : [{uncertainty.min().item():.8f}, {uncertainty.max().item():.8f}]')
                print(f"EDL-{getattr(model, 'edl_loss_type', 'mse').upper()} components : error={components['error'].item():.8f}, variance={components['variance'].item():.8f}, kl={components['kl'].item():.8f}, kl_coef={components['annealing_coef']:.4f},zero_evidence_ratio={components['zero_evidence_ratio'].item():.6f}")
                print(f"PASS: EDL input, probability, uncertainty, EDL-{getattr(model, 'edl_loss_type', 'mse').upper()} and KL are active.\n")
                model._edl_training_verified = True
        else:
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
            loss1 = binary_cross_entropy(y.double(), t.double())
            if model.use_CL:
                loss1 = loss1 + model.cl_weight * ys[1]
            if model.emb_type.find('predcurc') != -1:
                if model.emb_type.find('his') != -1:
                    loss = model.l1 * loss1 + model.l2 * ys[1] + model.l3 * ys[2]
                else:
                    loss = model.l1 * loss1 + model.l2 * ys[1]
            elif model.emb_type.find('predhis') != -1:
                loss = model.l1 * loss1 + model.l2 * ys[1]
            else:
                loss = loss1
    elif model_name in ['dkt', 'dkvmn']:
        if getattr(model, 'use_edl', False):
            logits = ys[1]
            if logits.ndim != 3 or logits.size(-1) != model.num_classes:
                raise ValueError(f'{model_name.upper()} EDL selected logits must have shape [B,T,{model.num_classes}], got {tuple(logits.shape)}.')
            if sm.dtype != torch.bool:
                raise TypeError(f'{model_name.upper()} EDL smask must be bool, got {sm.dtype}.')
            if logits.shape[:2] != sm.shape:
                raise ValueError(f'{model_name.upper()} EDL logits and smask are not aligned: logits={logits.shape[:2]}, smask={sm.shape}.')
            if rshft.shape != sm.shape:
                raise ValueError(f'{model_name.upper()} EDL target and smask are not aligned: target={rshft.shape}, smask={sm.shape}.')
            valid_logits = logits[sm]
            valid_targets = torch.masked_select(rshft, sm).long()
            if valid_logits.size(0) != valid_targets.size(0):
                raise RuntimeError(f'{model_name.upper()} EDL logits and targets have different sample counts.')
            if valid_targets.numel() == 0:
                raise ValueError(f'No valid {model_name.upper()} EDL samples remain after smask.')
            if not torch.all((valid_targets == 0) | (valid_targets == 1)):
                raise ValueError(f'{model_name.upper()} EDL targets must contain only 0 or 1.')
            target_one_hot = one_hot(valid_targets, num_classes=model.num_classes).to(device=valid_logits.device, dtype=valid_logits.dtype)
            edl_loss, components = edl_mse_loss(logits=valid_logits, target_one_hot=target_one_hot, epoch_num=epoch_num, annealing_step=model.edl_annealing_step, num_classes=model.num_classes, return_components=True, loss_type=getattr(model, 'edl_loss_type', 'mse'))
            loss = edl_loss
            model._last_edl_stats = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in components.items()}
            model._last_edl_stats['cl_loss'] = 0.0
            model._last_edl_stats['total_with_cl'] = float(loss.detach().cpu())
            if not getattr(model, '_edl_training_verified', False):
                with torch.no_grad():
                    _, alpha, probabilities, uncertainty = dirichlet_from_logits(valid_logits, num_classes=model.num_classes)
                    probability_sums = probabilities.sum(dim=-1)
                    assert torch.isfinite(probabilities).all()
                    assert torch.all(alpha >= 1.0)
                    assert torch.all(uncertainty > 0.0)
                    assert torch.all(uncertainty <= 1.0)
                    assert torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=1e-06, rtol=1e-06)
                print(f'\n[{model_name.upper()}+EDL complete training-path verification]')
                print(f'valid logits shape : {tuple(valid_logits.shape)}')
                print(f'probability range  : [{probabilities.min().item():.8f}, {probabilities.max().item():.8f}]')
                print(f'probability sum    : [{probability_sums.min().item():.8f}, {probability_sums.max().item():.8f}]')
                print(f'uncertainty range  : [{uncertainty.min().item():.8f}, {uncertainty.max().item():.8f}]')
                print(f'PASS: {model_name.upper()}+EDL loss {edl_loss.item():.8f}')
                print(f'PASS: {model_name.upper()}+EDL loss is active.\n')
                model._edl_training_verified = True
        else:
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
            loss = binary_cross_entropy(y.double(), t.double())
    elif model_name in ['akt', 'robustkt']:
        if getattr(model, 'use_edl', False):
            logits = ys[1]
            if logits.ndim != 3 or logits.size(-1) != model.num_classes:
                raise ValueError(f'{model_name.upper()} EDL logits must have shape [B,T,{model.num_classes}], got {tuple(logits.shape)}.')
            if sm.dtype != torch.bool:
                raise TypeError(f'{model_name.upper()} EDL smask must be bool, got {sm.dtype}.')
            if logits.shape[:2] != sm.shape:
                raise ValueError(f'{model_name.upper()} EDL logits and smask are not aligned: logits={logits.shape[:2]}, smask={sm.shape}.')
            if rshft.shape != sm.shape:
                raise ValueError(f'{model_name.upper()} EDL targets and smask are not aligned: targets={rshft.shape}, smask={sm.shape}.')
            valid_logits = logits[sm]
            valid_targets = torch.masked_select(rshft, sm).long()
            if valid_logits.size(0) != valid_targets.size(0):
                raise RuntimeError(f'{model_name.upper()} EDL logits and targets have different sample counts.')
            if valid_targets.numel() == 0:
                raise ValueError(f'No valid {model_name.upper()} EDL samples remain after smask.')
            if not torch.all((valid_targets == 0) | (valid_targets == 1)):
                raise ValueError(f'{model_name.upper()} EDL targets must contain only 0 or 1.')
            target_one_hot = one_hot(valid_targets, num_classes=model.num_classes).to(device=valid_logits.device, dtype=valid_logits.dtype)
            edl_loss, components = edl_mse_loss(logits=valid_logits, target_one_hot=target_one_hot, epoch_num=epoch_num, annealing_step=model.edl_annealing_step, num_classes=model.num_classes, return_components=True, loss_type=getattr(model, 'edl_loss_type', 'mse'))
            if len(preloss) == 0:
                raise RuntimeError(f'{model_name.upper()} regularization loss is missing.')
            reg_loss = preloss[0]
            loss = edl_loss + reg_loss
            model._last_edl_stats = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in components.items()}
            model._last_edl_stats['cl_loss'] = 0.0
            model._last_edl_stats['total_with_cl'] = float(loss.detach().cpu())
            if not getattr(model, '_edl_training_verified', False):
                with torch.no_grad():
                    _, alpha, probabilities, uncertainty = dirichlet_from_logits(valid_logits, num_classes=model.num_classes)
                    probability_sums = probabilities.sum(dim=-1)
                    assert torch.isfinite(probabilities).all()
                    assert torch.all(alpha >= 1.0)
                    assert torch.all(uncertainty > 0.0)
                    assert torch.all(uncertainty <= 1.0)
                    assert torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=1e-06, rtol=1e-06)
                print(f'\n[{model_name.upper()}+EDL complete training-path verification]')
                print(f'valid logits shape : {tuple(valid_logits.shape)}')
                print(f'probability range  : [{probabilities.min().item():.8f}, {probabilities.max().item():.8f}]')
                print(f'probability sum    : [{probability_sums.min().item():.8f}, {probability_sums.max().item():.8f}]')
                print(f'uncertainty range  : [{uncertainty.min().item():.8f}, {uncertainty.max().item():.8f}]')
                print(f'EDL loss          : {edl_loss.item():.8f}')
                print(f'{model_name.upper()} regularization: {float(reg_loss):.8f}')
                print(f'PASS: {model_name.upper()}+EDL loss is active.\n')
                model._edl_training_verified = True
        else:
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
            loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    return loss

def model_forward(model, data, rel=None, epoch_num=0):
    model_name = model.model_name
    dcur = data
    q, c, r, t = (dcur['qseqs'].to(device), dcur['cseqs'].to(device), dcur['rseqs'].to(device), dcur['tseqs'].to(device))
    qshft, cshft, rshft, tshft = (dcur['shft_qseqs'].to(device), dcur['shft_cseqs'].to(device), dcur['shft_rseqs'].to(device), dcur['shft_tseqs'].to(device))
    m, sm = (dcur['masks'].to(device), dcur['smasks'].to(device))
    ys, preloss = ([], [])
    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)
    if model_name == 'atdkt':
        if getattr(model, 'use_edl', False):
            all_probabilities, all_logits, y2, y3 = model(dcur, train=True)
            if all_probabilities.ndim != 3 or all_probabilities.size(-1) != model.num_c:
                raise ValueError(f'ATDKT+EDL probabilities must have shape [B,T,{model.num_c}], got {tuple(all_probabilities.shape)}.')
            if all_logits.ndim != 4 or all_logits.size(-2) != model.num_c or all_logits.size(-1) != model.num_classes:
                raise ValueError(f'ATDKT+EDL logits must have shape [B,T,{model.num_c},{model.num_classes}], got {tuple(all_logits.shape)}.')
            concept_one_hot = one_hot(cshft.long(), num_classes=model.num_c)
            y = (all_probabilities * concept_one_hot.to(dtype=all_probabilities.dtype)).sum(dim=-1)
            selected_logits = (all_logits * concept_one_hot.unsqueeze(-1).to(dtype=all_logits.dtype)).sum(dim=-2)
            ys = [y, selected_logits, y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            if model.emb_type.find('bkt') == -1 and model.emb_type.find('addcshft') == -1:
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            ys = [y, y2, y3]
    elif model_name in ['simplekt', 'keenkt']:
        if getattr(model, 'use_edl', False):
            y, logits, y2, y3 = model(dcur, train=True)
            ys = [y[:, 1:], logits[:, 1:, :], y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:, 1:], y2, y3]
    elif model_name == 'ukt':
        if model.use_edl:
            if model.use_CL:
                y, logits, sim, y2, y3, temp = model(dcur, train=True)
                ys = [y[:, 1:], logits[:, 1:, :]]
                ys.extend([sim, y2, y3])
            else:
                y, logits, y2, y3 = model(dcur, train=True)
                ys = [y[:, 1:], logits[:, 1:, :]]
                ys.extend([y2, y3])
        elif model.use_CL:
            y, sim, y2, y3, temp = model(dcur, train=True)
            ys = [y[:, 1:], sim, y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:, 1:], y2, y3]
    if model_name == 'dkt':
        if getattr(model, 'use_edl', False):
            all_probabilities, edl_output = model(c.long(), r.long(), return_edl=True)
            all_logits = edl_output['logits']
            if all_logits.ndim != 4 or all_logits.size(2) != model.num_c or all_logits.size(3) != model.num_classes:
                raise ValueError(f'DKT EDL logits must have shape [B,T,num_c,num_classes], got {tuple(all_logits.shape)}.')
            concept_one_hot = one_hot(cshft.long(), num_classes=model.num_c)
            selected_probabilities = (all_probabilities * concept_one_hot).sum(dim=-1)
            selected_logits = (all_logits * concept_one_hot.unsqueeze(-1).to(dtype=all_logits.dtype)).sum(dim=2)
            ys = [selected_probabilities, selected_logits]
        else:
            all_probabilities = model(c.long(), r.long())
            concept_one_hot = one_hot(cshft.long(), num_classes=model.num_c)
            selected_probabilities = (all_probabilities * concept_one_hot).sum(dim=-1)
            ys.append(selected_probabilities)
    elif model_name == 'dkvmn':
        if getattr(model, 'use_edl', False):
            probabilities, edl_output = model(cc.long(), cr.long(), return_edl=True)
            selected_probabilities = probabilities[:, 1:]
            selected_logits = edl_output['logits'][:, 1:]
            ys = [selected_probabilities, selected_logits]
        else:
            y = model(cc.long(), cr.long())
            ys.append(y[:, 1:])
    elif model_name in ['akt', 'robustkt']:
        if getattr(model, 'use_edl', False):
            y, reg_loss, edl_output = model(cc.long(), cr.long(), cq.long(), return_edl=True)
            ys = [y[:, 1:], edl_output['logits'][:, 1:, :]]
            preloss.append(reg_loss)
        else:
            y, reg_loss = model(cc.long(), cr.long(), cq.long())
            ys.append(y[:, 1:])
            preloss.append(reg_loss)
    loss = cal_loss(model, ys, r, rshft, sm, preloss=preloss, epoch_num=epoch_num)
    if model_name == 'ukt' and model.use_CL != 0:
        return (loss, temp)
    return loss

def pareto_normalized_ideal_checkpoint(candidates):
    if not candidates:
        return (None, float('inf'), [])
    pareto_front = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            no_worse = other['u_correct'] <= candidate['u_correct'] and other['u_gap'] >= candidate['u_gap']
            strictly_better = other['u_correct'] < candidate['u_correct'] or other['u_gap'] > candidate['u_gap']
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_front.append(candidate)
    u_correct_values = [candidate['u_correct'] for candidate in candidates]
    u_gap_values = [candidate['u_gap'] for candidate in candidates]
    u_correct_min = min(u_correct_values)
    u_correct_max = max(u_correct_values)
    u_gap_min = min(u_gap_values)
    u_gap_max = max(u_gap_values)
    u_correct_range = max(u_correct_max - u_correct_min, 1e-12)
    u_gap_range = max(u_gap_max - u_gap_min, 1e-12)
    best_candidate = None
    best_distance = float('inf')
    for candidate in pareto_front:
        u_correct_norm = (candidate['u_correct'] - u_correct_min) / u_correct_range
        u_gap_norm = (candidate['u_gap'] - u_gap_min) / u_gap_range
        distance = (u_correct_norm ** 2 + (1.0 - u_gap_norm) ** 2) ** 0.5
        distance = float(distance)
        if distance < best_distance or (distance == best_distance and best_candidate is not None and (candidate['auc'] > best_candidate['auc'])):
            best_candidate = candidate
            best_distance = distance
    return (best_candidate, best_distance, pareto_front)

def print_edl_eval_stats(prefix, stats):
    if stats is None:
        return
    print(f"            {prefix}: AUC={stats['auc']:.4g}, ACC={stats['acc']:.4g}, MUD={stats['mud']:.4g}, W2={stats['w2']:.4g}")

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None):
    max_auc = 0
    best_epoch = -1
    best_ideal_distance = float('inf')
    last_checkpoint_reason = 'none'
    checkpoint_candidates = []
    pareto_front_size = 0
    current_pareto_best_epoch = -1
    best_checkpoint_stats = {'auc': -1.0, 'acc': -1.0, 'u_correct': -1.0, 'u_wrong': -1.0, 'u_gap': -1.0, 'ideal_distance': float('inf')}
    candidate_ckpt_dir = os.path.join(ckpt_path, 'pareto_candidates')
    train_step = 0
    use_uncertainty_checkpoint = bool(getattr(model, 'use_uncertainty_checkpoint', False))
    checkpoint_auc_tolerance = float(getattr(model, 'checkpoint_auc_tolerance', 0.001))
    early_stop_patience = int(getattr(model, 'early_stop_patience', 10))
    early_stop_patience = max(early_stop_patience, 1)
    testauc = -1
    testacc = -1
    window_testauc = -1
    window_testacc = -1
    validauc = -1
    validacc = -1
    rel = None
    scheduler = None
    for i in range(1, num_epochs + 1):
        epoch_index = i - 1
        loss_mean = []
        edl_epoch_sums = {'total': 0.0, 'error': 0.0, 'variance': 0.0, 'kl': 0.0, 'uncertainty_mean': 0.0, 'strength_mean': 0.0, 'evidence_mean': 0.0, 'zero_evidence_ratio': 0.0}
        edl_epoch_valid_count = 0
        edl_epoch_batch_count = 0
        edl_epoch_cl_sum = 0.0
        edl_epoch_annealing_coef = 0.0
        for step_in_epoch, data in enumerate(train_loader, start=1):
            train_step += 1
            if model.model_name == 'ukt':
                model.current_epoch = i
                model.current_step = step_in_epoch
                model.total_train_step = train_step
            model.train()
            if model.model_name == 'ukt' and model.use_CL:
                loss, temp = model_forward(model, data, epoch_num=epoch_index)
            else:
                loss = model_forward(model, data, epoch_num=epoch_index)
            opt.zero_grad()
            loss.backward()
            if model.model_name == 'ukt' and step_in_epoch == 1 and hasattr(model, 'print_cov_debug'):
                model.print_cov_debug()
            opt.step()
            if model.model_name in ['ukt', 'akt', 'robustkt', 'dkt', 'dkvmn', 'simplekt', 'keenkt', 'atdkt'] and getattr(model, 'use_edl', False) and hasattr(model, '_last_edl_stats'):
                batch_stats = model._last_edl_stats
                valid_count = int(batch_stats['valid_count'])
                if valid_count > 0:
                    edl_epoch_valid_count += valid_count
                    edl_epoch_batch_count += 1
                    for key in edl_epoch_sums:
                        edl_epoch_sums[key] += batch_stats.get(key, 0.0) * valid_count
                    edl_epoch_cl_sum += float(batch_stats.get('cl_loss', 0.0))
                    edl_epoch_annealing_coef = float(batch_stats.get('annealing_coef', edl_epoch_annealing_coef))
            loss_mean.append(loss.detach().cpu().numpy())
        if scheduler is not None:
            scheduler.step()
        loss_mean = float(np.mean(loss_mean))
        edl_epoch_stats = None
        if edl_epoch_valid_count > 0:
            edl_epoch_stats = {key: value / edl_epoch_valid_count for key, value in edl_epoch_sums.items()}
            edl_epoch_stats['valid_count'] = edl_epoch_valid_count
            edl_epoch_stats['batch_count'] = edl_epoch_batch_count
            edl_epoch_stats['annealing_coef'] = edl_epoch_annealing_coef
            edl_epoch_stats['cl_loss'] = edl_epoch_cl_sum / max(edl_epoch_batch_count, 1)
            edl_epoch_stats['total_with_cl'] = loss_mean
        valid_edl_stats = None
        if model.model_name == 'ukt' and getattr(model, 'use_edl', False):
            valid_edl_stats = evaluate_edl_stats(model, valid_loader)
            auc = valid_edl_stats['auc']
            acc = valid_edl_stats['acc']
        else:
            auc, acc = evaluate(model, valid_loader, model.model_name)
        current_u_wrong = valid_edl_stats.get('pred_wrong_uncertainty_mean', 0.0) if valid_edl_stats is not None else 0.0
        current_u_correct = valid_edl_stats.get('pred_correct_uncertainty_mean', 0.0) if valid_edl_stats is not None else 0.0
        current_u_gap = valid_edl_stats.get('pred_uncertainty_gap', 0.0) if valid_edl_stats is not None else 0.0
        current_ideal_distance = float('inf')
        should_checkpoint = False
        checkpoint_reason = 'none'
        if use_uncertainty_checkpoint and model.model_name == 'ukt' and getattr(model, 'use_edl', False):
            auc_improved = auc > max_auc
            reference_auc = max(max_auc, auc)
            auc_within_tolerance = auc >= reference_auc - checkpoint_auc_tolerance
            if auc_within_tolerance:
                candidate_ckpt_path = None
                if save_model:
                    os.makedirs(candidate_ckpt_dir, exist_ok=True)
                    candidate_ckpt_path = os.path.join(candidate_ckpt_dir, f'{model.emb_type}_epoch_{i}.ckpt')
                    torch.save(model.state_dict(), candidate_ckpt_path)
                checkpoint_candidates.append({'epoch': i, 'auc': float(auc), 'acc': float(acc), 'u_correct': float(current_u_correct), 'u_wrong': float(current_u_wrong), 'u_gap': float(current_u_gap), 'ckpt_path': candidate_ckpt_path})
            auc_floor = reference_auc - checkpoint_auc_tolerance
            checkpoint_candidates = [candidate for candidate in checkpoint_candidates if candidate['auc'] >= auc_floor]
            pareto_best_candidate, pareto_best_distance, pareto_front = pareto_normalized_ideal_checkpoint(checkpoint_candidates)
            pareto_front_size = len(pareto_front)
            current_pareto_best_epoch = pareto_best_candidate['epoch'] if pareto_best_candidate is not None else -1
            current_ideal_distance = pareto_best_distance
            if auc_improved:
                should_checkpoint = True
                checkpoint_reason = 'auc'
            elif auc_within_tolerance and current_pareto_best_epoch == i and (best_epoch != i):
                should_checkpoint = True
                checkpoint_reason = 'pareto_ideal'
        elif auc > max_auc + 0.001:
            should_checkpoint = True
            checkpoint_reason = 'auc'
        if should_checkpoint:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + '_model.ckpt'))
            if auc > max_auc:
                max_auc = auc
            best_ideal_distance = current_ideal_distance
            last_checkpoint_reason = checkpoint_reason
            best_checkpoint_stats = {'auc': float(auc), 'acc': float(acc), 'u_correct': float(current_u_correct), 'u_wrong': float(current_u_wrong), 'u_gap': float(current_u_gap), 'ideal_distance': float(current_ideal_distance)}
            best_epoch = i
            if use_uncertainty_checkpoint and model.model_name == 'ukt' and getattr(model, 'use_edl', False):
                print(f'            Saved checkpoint: epoch={best_epoch}, reason={checkpoint_reason}, validauc={auc:.4g}, validacc={acc:.4g}, u_correct={current_u_correct:.4g}, u_wrong={current_u_wrong:.4g}, u_gap={current_u_gap:.4g}, ideal_distance={current_ideal_distance:.4g}, candidate_count={len(checkpoint_candidates)}, pareto_front_size={pareto_front_size}')
            testauc = -1
            testacc = -1
            window_testauc = -1
            window_testacc = -1
            if not save_model:
                if test_loader is not None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + '_test_predictions.txt')
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_path=save_test_path)
                if test_window_loader is not None:
                    save_window_path = os.path.join(ckpt_path, model.emb_type + '_test_window_predictions.txt')
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_path=save_window_path)
            validauc = auc
            validacc = acc
        print(f'Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}')
        if use_uncertainty_checkpoint and model.model_name == 'ukt' and getattr(model, 'use_edl', False):
            print(f"            Checkpoint selection: best_epoch={best_epoch}, current_auc={auc:.4g}, max_auc={max_auc:.4g}, auc_tolerance={checkpoint_auc_tolerance:.4g}, current_u_correct={current_u_correct:.4g}, current_u_wrong={current_u_wrong:.4g}, current_u_gap={current_u_gap:.4g}, pareto_ideal_distance={current_ideal_distance:.4g}, candidate_count={len(checkpoint_candidates)}, pareto_front_size={pareto_front_size}, pareto_best_epoch={current_pareto_best_epoch}, best_auc={best_checkpoint_stats['auc']:.4g}, best_u_correct={best_checkpoint_stats['u_correct']:.4g}, best_u_wrong={best_checkpoint_stats['u_wrong']:.4g}, best_u_gap={best_checkpoint_stats['u_gap']:.4g}, best_ideal_distance={best_ideal_distance:.4g}, last_reason={last_checkpoint_reason}, patience={early_stop_patience}")
        if model.model_name in ['ukt', 'akt', 'robustkt', 'dkt', 'dkvmn', 'simplekt', 'keenkt', 'atdkt'] and getattr(model, 'use_edl', False) and (edl_epoch_stats is not None):
            stats = edl_epoch_stats
            print(f"            {model.model_name.upper()}+EDL epoch: total={stats['total']:.6f}, total_with_cl={stats['total_with_cl']:.6f}, error={stats['error']:.6f}, variance={stats['variance']:.6f}, kl={stats['kl']:.6f}, kl_coef={stats['annealing_coef']:.3f}, uncertainty={stats['uncertainty_mean']:.6f}, strength={stats['strength_mean']:.6f}, evidence={stats['evidence_mean']:.6f}, zero_evidence={stats['zero_evidence_ratio']:.6f}, cl_loss={stats['cl_loss']:.6f}, valid_count={stats['valid_count']}")
        if i - best_epoch >= early_stop_patience:
            break
    if use_uncertainty_checkpoint and model.model_name == 'ukt' and getattr(model, 'use_edl', False) and checkpoint_candidates:
        final_candidate, final_distance, final_pareto_front = pareto_normalized_ideal_checkpoint(checkpoint_candidates)
        if final_candidate is not None:
            final_ckpt_path = final_candidate.get('ckpt_path')
            if save_model and final_ckpt_path and os.path.isfile(final_ckpt_path):
                net = torch.load(final_ckpt_path, map_location=device)
                model.load_state_dict(net)
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + '_model.ckpt'))
            best_epoch = int(final_candidate['epoch'])
            validauc = float(final_candidate['auc'])
            validacc = float(final_candidate['acc'])
            best_ideal_distance = float(final_distance)
            best_checkpoint_stats = {'auc': validauc, 'acc': validacc, 'u_correct': float(final_candidate['u_correct']), 'u_wrong': float(final_candidate['u_wrong']), 'u_gap': float(final_candidate['u_gap']), 'ideal_distance': best_ideal_distance}
            print(f"Final Pareto checkpoint: epoch={best_epoch}, validauc={validauc:.4g}, validacc={validacc:.4g}, u_correct={best_checkpoint_stats['u_correct']:.4g}, u_wrong={best_checkpoint_stats['u_wrong']:.4g}, u_gap={best_checkpoint_stats['u_gap']:.4g}, ideal_distance={best_ideal_distance:.4g}, candidate_count={len(checkpoint_candidates)}, pareto_front_size={len(final_pareto_front)}")
    return (testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch)
