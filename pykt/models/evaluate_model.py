import numpy as np
import torch
from torch.nn.functional import one_hot
from sklearn import metrics
from .evaluate_uncertainty import calculate_mean_uncertainty
from .edl import dirichlet_from_logits
import pandas as pd
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

def evaluate_edl(model, test_loader):
    """
    8 个模型共用的 EDL 知识点级评估。

    支持 DKT、AKT、AT-DKT、DKVMN、UKT、KeenKT、simpleKT、RobustKT。

    逐样本数据仅在内存中用于计算评估指标，不导出文件。

    result_df每一行对应一个smask有效位置，包含：
        concept_id
        true_label
        pred_probability
        uncertainty
        pred_label
        is_error
    """
    if not getattr(model, 'use_edl', False):
        raise ValueError('evaluate_edl requires model.use_edl=True.')
    model.eval()
    all_targets = []
    all_probabilities = []
    all_uncertainties = []
    all_concept_ids = []
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, (tuple, list)):
                dcur = data[0]
            else:
                dcur = data
            dcur = {key: value.to(device) if torch.is_tensor(value) else value for key, value in dcur.items()}
            rshft = dcur['shft_rseqs'].long()
            cshft = dcur['shft_cseqs'].long()
            sm = dcur['smasks'].bool()
            if model.model_name in ['ukt']:
                probabilities, edl_output = model(dcur, return_edl=True)
            elif model.model_name in ['simplekt', 'keenkt']:
                probabilities, edl_output = model(dcur, return_edl=True)
            elif model.model_name in ['akt', 'robustkt']:
                q = dcur['qseqs'].long()
                c = dcur['cseqs'].long()
                r = dcur['rseqs'].long()
                qshft = dcur['shft_qseqs'].long()
                cshft_for_model = dcur['shft_cseqs'].long()
                rshft_for_model = dcur['shft_rseqs'].long()
                cq = torch.cat((q[:, 0:1], qshft), dim=1)
                cc = torch.cat((c[:, 0:1], cshft_for_model), dim=1)
                cr = torch.cat((r[:, 0:1], rshft_for_model), dim=1)
                probabilities, _, edl_output = model(cc, cr, cq, return_edl=True)
            elif model.model_name == 'dkt':
                c = dcur['cseqs'].long()
                r = dcur['rseqs'].long()
                dkt_all_probabilities, edl_output = model(c, r, return_edl=True)
                dkt_all_uncertainties = edl_output['uncertainty']
                if dkt_all_probabilities.ndim != 3 or dkt_all_probabilities.size(-1) != model.num_c:
                    raise ValueError(f'DKT EDL probabilities must have shape [B,T,num_c], got {tuple(dkt_all_probabilities.shape)}.')
                if dkt_all_uncertainties.shape != dkt_all_probabilities.shape:
                    raise ValueError(f'DKT EDL probability and uncertainty shapes differ: {dkt_all_probabilities.shape} vs {dkt_all_uncertainties.shape}.')
                concept_one_hot = one_hot(cshft.long(), num_classes=model.num_c)
                probabilities = (dkt_all_probabilities * concept_one_hot).sum(dim=-1)
                uncertainties = (dkt_all_uncertainties * concept_one_hot.to(dtype=dkt_all_uncertainties.dtype)).sum(dim=-1)
            elif model.model_name == 'atdkt':
                atdkt_all_probabilities, edl_output = model(dcur, return_edl=True)
                atdkt_all_uncertainties = edl_output['uncertainty']
                if atdkt_all_probabilities.ndim != 3 or atdkt_all_probabilities.size(-1) != model.num_c:
                    raise ValueError(f'ATDKT EDL probabilities must have shape [B,T,num_c], got {tuple(atdkt_all_probabilities.shape)}.')
                if atdkt_all_uncertainties.shape != atdkt_all_probabilities.shape:
                    raise ValueError(f'ATDKT EDL probability and uncertainty shapes differ: {atdkt_all_probabilities.shape} vs {atdkt_all_uncertainties.shape}.')
                if edl_output['logits'].ndim != 4 or edl_output['logits'].size(-2) != model.num_c or edl_output['logits'].size(-1) != model.num_classes:
                    raise ValueError(f"ATDKT EDL logits must have shape [B,T,{model.num_c},{model.num_classes}], got {tuple(edl_output['logits'].shape)}.")
                concept_one_hot = one_hot(cshft.long(), num_classes=model.num_c)
                probabilities = (atdkt_all_probabilities * concept_one_hot.to(dtype=atdkt_all_probabilities.dtype)).sum(dim=-1)
                uncertainties = (atdkt_all_uncertainties * concept_one_hot.to(dtype=atdkt_all_uncertainties.dtype)).sum(dim=-1)
            elif model.model_name == 'dkvmn':
                c = dcur['cseqs'].long()
                r = dcur['rseqs'].long()
                cshft_for_model = dcur['shft_cseqs'].long()
                rshft_for_model = dcur['shft_rseqs'].long()
                cc = torch.cat((c[:, 0:1], cshft_for_model), dim=1)
                cr = torch.cat((r[:, 0:1], rshft_for_model), dim=1)
                probabilities, edl_output = model(cc, cr, return_edl=True)
                probabilities = probabilities[:, 1:]
                uncertainties = edl_output['uncertainty'][:, 1:]
            else:
                raise ValueError(f'evaluate_edl currently supports only UKT, AKT, RobustKT, DKT, DKVMN, ATDKT, SimpleKT and KeenKT, got {model.model_name}.')
            if model.model_name in ['ukt', 'akt', 'robustkt', 'simplekt', 'keenkt']:
                uncertainties = edl_output['uncertainty']
                probabilities = probabilities[:, 1:]
                uncertainties = uncertainties[:, 1:]
            elif model.model_name in ['dkt', 'dkvmn', 'atdkt']:
                pass
            if probabilities.shape != sm.shape:
                raise ValueError(f'EDL probability and smask shapes do not match: {probabilities.shape} vs {sm.shape}.')
            if uncertainties.shape != sm.shape:
                raise ValueError(f'EDL uncertainty and smask shapes do not match: {uncertainties.shape} vs {sm.shape}.')
            if rshft.shape != sm.shape:
                raise ValueError(f'EDL target and smask shapes do not match: {rshft.shape} vs {sm.shape}.')
            if cshft.shape != sm.shape:
                raise ValueError(f'EDL concept_id and smask shapes do not match: {cshft.shape} vs {sm.shape}.')
            valid_targets = rshft[sm]
            valid_probabilities = probabilities[sm]
            valid_uncertainties = uncertainties[sm]
            valid_concept_ids = cshft[sm]
            if not valid_targets.shape == valid_probabilities.shape == valid_uncertainties.shape == valid_concept_ids.shape:
                raise RuntimeError('EDL targets, probabilities, uncertainties and concept_ids are misaligned.')
            all_targets.append(valid_targets.detach().cpu())
            all_probabilities.append(valid_probabilities.detach().cpu())
            all_uncertainties.append(valid_uncertainties.detach().cpu())
            all_concept_ids.append(valid_concept_ids.detach().cpu())
    if len(all_targets) == 0:
        raise ValueError('The test loader produced no EDL samples.')
    targets = torch.cat(all_targets, dim=0).numpy()
    probabilities = torch.cat(all_probabilities, dim=0).numpy()
    uncertainties = torch.cat(all_uncertainties, dim=0).numpy()
    concept_ids = torch.cat(all_concept_ids, dim=0).numpy()
    pred_labels = (probabilities >= 0.5).astype(np.int64)
    is_error = (pred_labels != targets).astype(np.int64)
    if np.unique(targets).size >= 2:
        auc = metrics.roc_auc_score(y_true=targets, y_score=probabilities)
    else:
        auc = float('nan')
    acc = metrics.accuracy_score(targets, pred_labels)
    result_df = pd.DataFrame({'concept_id': concept_ids.astype(np.int64), 'true_label': targets.astype(np.int64), 'pred_probability': probabilities, 'uncertainty': uncertainties, 'pred_label': pred_labels, 'is_error': is_error})
    mean_result = calculate_mean_uncertainty(result_df)
    print(f"EDL evaluation: samples={len(result_df)}, auc={auc:.6f}, acc={acc:.6f}, MUD={mean_result['mean_u_gap']:.6f}, W2={mean_result['w2']:.6f}")
    return (auc, acc, result_df)

def evaluate_edl_stats(model, test_loader):
    """Return the four metrics plus internal checkpoint statistics."""
    auc, acc, result_df = evaluate_edl(model, test_loader)
    mean_result = calculate_mean_uncertainty(result_df)
    correct_mean = mean_result['mean_u_correct']
    wrong_mean = mean_result['mean_u_wrong']
    correct_mean = correct_mean if mean_result['correct_count'] else 0.0
    wrong_mean = wrong_mean if mean_result['wrong_count'] else 0.0
    return {'auc': float(auc), 'acc': float(acc), 'mud': mean_result['mean_u_gap'], 'w2': mean_result['w2'], 'samples': len(result_df), 'pred_correct_uncertainty_mean': correct_mean, 'pred_wrong_uncertainty_mean': wrong_mean, 'pred_uncertainty_gap': wrong_mean - correct_mean}

def save_cur_predict_result(dres, q, r, d, t, m, sm, p):
    results = []
    for i in range(0, t.shape[0]):
        cps = torch.masked_select(p[i], sm[i]).detach().cpu()
        cts = torch.masked_select(t[i], sm[i]).detach().cpu()
        cqs = torch.masked_select(q[i], m[i]).detach().cpu()
        crs = torch.masked_select(r[i], m[i]).detach().cpu()
        cds = torch.masked_select(d[i], sm[i]).detach().cpu()
        qs, rs, ts, ps, ds = ([], [], [], [], [])
        for cq, cr in zip(cqs.int(), crs.int()):
            qs.append(cq.item())
            rs.append(cr.item())
        for ct, cp, cd in zip(cts.int(), cps, cds.int()):
            ts.append(ct.item())
            ps.append(cp.item())
            ds.append(cd.item())
        try:
            auc = metrics.roc_auc_score(y_true=np.array(ts), y_score=np.array(ps))
        except Exception as e:
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        dres[len(dres)] = [qs, rs, ds, ts, ps, prelabels, auc, acc]
        results.append(str([qs, rs, ds, ts, ps, prelabels, auc, acc]))
    return '\n'.join(results)

def evaluate(model, test_loader, model_name, rel=None, save_path=''):
    if save_path != '':
        fout = open(save_path, 'w', encoding='utf8')
    with torch.no_grad():
        y_trues = []
        y_scores = []
        dres = dict()
        test_mini_index = 0
        for data in test_loader:
            dcur = data
            q, c, r = (dcur['qseqs'], dcur['cseqs'], dcur['rseqs'])
            qshft, cshft, rshft = (dcur['shft_qseqs'], dcur['shft_cseqs'], dcur['shft_rseqs'])
            m, sm = (dcur['masks'], dcur['smasks'])
            q, c, r, qshft, cshft, rshft, m, sm = (q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device))
            model.eval()
            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)
            if model_name in ['atdkt']:
                '\n                y = model(dcur) \n                import pickle\n                with open(f"{test_mini_index}_result.pkl",\'wb\') as f:\n                    data = {"y":y,"cshft":cshft,"num_c":model.num_c,"rshft":rshft,"qshft":qshft,"sm":sm}\n                    pickle.dump(data,f)\n                '
                y = model(dcur)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ['simplekt', 'keenkt', 'ukt']:
                y = model(dcur)
                y = y[:, 1:]
            elif model_name in ['dkt']:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ['dkvmn']:
                y = model(cc.long(), cr.long())
                y = y[:, 1:]
            elif model_name in ['akt', 'robustkt']:
                y, reg_loss = model(cc.long(), cr.long(), cq.long())
                y = y[:, 1:]
            if save_path != '':
                result = save_cur_predict_result(dres, c, r, cshft, rshft, m, sm, y)
                fout.write(result + '\n')
            y = torch.masked_select(y, sm).detach().cpu()
            t = torch.masked_select(rshft, sm).detach().cpu()
            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
            test_mini_index += 1
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f'ts.shape: {ts.shape}, ps.shape: {ps.shape}')
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
    return (auc, acc)

def early_fusion(curhs, model, model_name):
    if model_name in ['dkvmn']:
        p = model.p_layer(model.dropout_layer(curhs[0]))
        p = torch.sigmoid(p)
        p = p.squeeze(-1)
    elif model_name == 'ukt':
        raw_output = model.out(curhs[0])
        if getattr(model, 'use_edl', False):
            _, _, probabilities, _ = dirichlet_from_logits(logits=raw_output, num_classes=model.num_classes)
            p = probabilities[..., 1]
        else:
            p = torch.sigmoid(raw_output.squeeze(-1))
    elif model_name in ['simplekt', 'keenkt']:
        raw_output = model.out(curhs[0])
        if getattr(model, 'use_edl', False):
            if raw_output.ndim < 2 or raw_output.size(-1) != model.num_classes:
                raise ValueError(f'SimpleKT/KeenKT EDL question-level logits must end with num_classes, got {tuple(raw_output.shape)}.')
            _, _, probabilities, _ = dirichlet_from_logits(logits=raw_output, num_classes=model.num_classes)
            p = probabilities[..., 1]
        else:
            p = torch.sigmoid(raw_output.squeeze(-1))
    elif model_name in ['akt', 'robustkt']:
        output = model.out(curhs[0]).squeeze(-1)
        p = torch.sigmoid(output)
    return p

def late_fusion(dcur, curdf, fusion_type=['mean', 'vote', 'all']):
    high, low = ([], [])
    for pred in curdf['preds']:
        if pred >= 0.5:
            high.append(pred)
        else:
            low.append(pred)
    if 'mean' in fusion_type:
        dcur.setdefault('late_mean', [])
        dcur['late_mean'].append(round(float(curdf['preds'].mean()), 4))
    if 'vote' in fusion_type:
        dcur.setdefault('late_vote', [])
        correctnum = list(curdf['preds'] >= 0.5).count(True)
        late_vote = np.mean(high) if correctnum / len(curdf['preds']) >= 0.5 else np.mean(low)
        dcur['late_vote'].append(late_vote)
    if 'all' in fusion_type:
        dcur.setdefault('late_all', [])
        late_all = np.mean(high) if correctnum == len(curdf['preds']) else np.mean(low)
        dcur['late_all'].append(late_all)
    return

def effective_fusion(df, model, model_name, fusion_type):
    dres = dict()
    df = df.groupby('qidx', as_index=True, sort=True)
    curhs, curr = ([[], []], [])
    dcur = {'late_trues': [], 'qidxs': [], 'questions': [], 'concepts': [], 'row': [], 'concept_preds': []}
    hasearly = ['dkvmn', 'akt', 'robustkt', 'simplekt', 'keenkt', 'ukt']
    for ui in df:
        curdf = ui[1]
        if model_name in hasearly:
            curhs[0].append(curdf['hidden'].mean().astype(float))
        else:
            pass
        curr.append(int(curdf['response'].mean()))
        dcur['late_trues'].append(int(curdf['response'].mean()))
        dcur['qidxs'].append(ui[0])
        dcur['row'].append(int(curdf['row'].mean()))
        dcur['questions'].append(','.join([str(int(s)) for s in curdf['questions'].tolist()]))
        dcur['concepts'].append(','.join([str(int(s)) for s in curdf['concepts'].tolist()]))
        late_fusion(dcur, curdf)
        dcur['concept_preds'].append(','.join([str(round(s, 4)) for s in curdf['preds'].tolist()]))
    for key in dcur:
        dres.setdefault(key, [])
        dres[key].append(np.array(dcur[key]))
    if 'early_fusion' in fusion_type and model_name in hasearly:
        curhs = [torch.tensor(curh).float().to(device) for curh in curhs]
        curr = torch.tensor(curr).long().to(device)
        p = early_fusion(curhs, model, model_name)
        dres.setdefault('early_trues', [])
        dres['early_trues'].append(curr.cpu().numpy())
        dres.setdefault('early_preds', [])
        dres['early_preds'].append(p.cpu().numpy())
    return dres

def group_fusion(dmerge, model, model_name, fusion_type, fout):
    hs, sms, cq, cc, rs, ps, qidxs, rests, orirows = (dmerge['hs'], dmerge['sm'], dmerge['cq'], dmerge['cc'], dmerge['cr'], dmerge['y'], dmerge['qidxs'], dmerge['rests'], dmerge['orirow'])
    if cq.shape[1] == 0:
        cq = cc
    hasearly = ['dkvmn', 'akt', 'robustkt', 'simplekt', 'keenkt', 'ukt']
    alldfs, drest = ([], dict())
    realbz = rs.shape[0]
    for bz in range(rs.shape[0]):
        cursm = [0] + sms[bz].cpu().tolist()
        curqidxs = [-1] + qidxs[bz].cpu().tolist()
        currests = [-1] + rests[bz].cpu().tolist()
        currows = [-1] + orirows[bz].cpu().tolist()
        curps = [-1] + ps[bz].cpu().tolist()
        df = pd.DataFrame({'qidx': curqidxs, 'rest': currests, 'row': currows, 'select': cursm, 'questions': cq[bz].cpu().tolist(), 'concepts': cc[bz].cpu().tolist(), 'response': rs[bz].cpu().tolist(), 'preds': curps})
        if model_name in hasearly:
            df['hidden'] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
        df = df[df['select'] != 0]
        alldfs.append(df)
    effective_dfs, rest_start = ([], -1)
    flag = False
    for i in range(len(alldfs) - 1, -1, -1):
        df = alldfs[i]
        counts = (df['rest'] == 0).value_counts()
        if not flag and False not in counts:
            flag = True
            effective_dfs.append(df)
            rest_start = i + 1
        elif flag:
            effective_dfs.append(df)
    if rest_start == -1:
        rest_start = 0
    for key in dmerge.keys():
        if key == 'hs':
            drest[key] = []
            if model_name in hasearly:
                drest[key] = [dmerge[key][0][rest_start:]]
        else:
            drest[key] = dmerge[key][rest_start:]
    restlen = drest['cr'].shape[0]
    dfs = dict()
    for df in effective_dfs:
        for i, row in df.iterrows():
            for key in row.keys():
                dfs.setdefault(key, [])
                dfs[key].extend([row[key]])
    df = pd.DataFrame(dfs)
    if df.shape[0] == 0:
        return ({}, drest)
    dres = effective_fusion(df, model, model_name, fusion_type)
    dfinal = dict()
    for key in dres:
        dfinal[key] = np.concatenate(dres[key], axis=0)
    early = False
    if model_name in hasearly and 'early_fusion' in fusion_type:
        early = True
    save_question_res(dfinal, fout, early)
    return (dfinal, drest)

def save_question_res(dres, fout, early=False):
    for i in range(0, len(dres['qidxs'])):
        row, qidx, qs, cs, lt, lm, lv, la = (dres['row'][i], dres['qidxs'][i], dres['questions'][i], dres['concepts'][i], dres['late_trues'][i], dres['late_mean'][i], dres['late_vote'][i], dres['late_all'][i])
        conceptps = dres['concept_preds'][i]
        curres = [row, qidx, qs, cs, conceptps, lt, lm, lv, la]
        if early:
            et, ep = (dres['early_trues'][i], dres['early_preds'][i])
            curres = curres + [et, ep]
        curstr = '\t'.join([str(round(s, 4)) if type(s) == type(0.1) or type(s) == np.float32 else str(s) for s in curres])
        fout.write(curstr + '\n')

def evaluate_question(model, test_loader, model_name, fusion_type=['early_fusion', 'late_fusion'], save_path=''):
    hasearly = ['dkvmn', 'akt', 'robustkt', 'simplekt', 'keenkt', 'ukt']
    if save_path != '':
        fout = open(save_path, 'w', encoding='utf8')
        if model_name in hasearly:
            fout.write('\t'.join(['orirow', 'qidx', 'questions', 'concepts', 'concept_preds', 'late_trues', 'late_mean', 'late_vote', 'late_all', 'early_trues', 'early_preds']) + '\n')
        else:
            fout.write('\t'.join(['orirow', 'qidx', 'questions', 'concepts', 'concept_preds', 'late_trues', 'late_mean', 'late_vote', 'late_all']) + '\n')
    with torch.no_grad():
        dinfos = dict()
        dhistory = dict()
        history_keys = ['hs', 'sm', 'cq', 'cc', 'cr', 'y', 'qidxs', 'rests', 'orirow']
        y_trues, y_scores = ([], [])
        lenc = 0
        for data in test_loader:
            dcurori, dqtest = data
            q, c, r = (dcurori['qseqs'], dcurori['cseqs'], dcurori['rseqs'])
            qshft, cshft, rshft = (dcurori['shft_qseqs'], dcurori['shft_cseqs'], dcurori['shft_rseqs'])
            m, sm = (dcurori['masks'], dcurori['smasks'])
            q, c, r, qshft, cshft, rshft, m, sm = (q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device))
            qidxs, rests, orirow = (dqtest['qidxs'], dqtest['rests'], dqtest['orirow'])
            lenc += q.shape[0]
            model.eval()
            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)
            dcur = dict()
            if model_name in ['dkvmn']:
                y, h = model(cc.long(), cr.long(), True)
                y = y[:, 1:]
            elif model_name in ['simplekt', 'keenkt', 'ukt']:
                y, h = model(dcurori, qtest=True, train=False)
                y = y[:, 1:]
            elif model_name in ['akt', 'robustkt']:
                y, reg_loss, h = model(cc.long(), cr.long(), cq.long(), True)
                y = y[:, 1:]
            elif model_name in ['atdkt']:
                y = model(dcurori)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ['dkt']:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            concepty = torch.masked_select(y, sm).detach().cpu()
            conceptt = torch.masked_select(rshft, sm).detach().cpu()
            y_trues.append(conceptt.numpy())
            y_scores.append(concepty.numpy())
            hs = []
            if model_name in hasearly:
                hs = [h]
            dcur['hs'], dcur['sm'], dcur['cq'], dcur['cc'], dcur['cr'], dcur['y'], dcur['qidxs'], dcur['rests'], dcur['orirow'] = (hs, sm, cq, cc, cr, y, qidxs, rests, orirow)
            dmerge = dict()
            for key in history_keys:
                if len(dhistory) == 0:
                    dmerge[key] = dcur[key]
                elif key == 'hs':
                    dmerge[key] = []
                    if model_name in hasearly:
                        dmerge[key] = [torch.cat((dhistory[key][0], dcur[key][0]), dim=0)]
                else:
                    dmerge[key] = torch.cat((dhistory[key], dcur[key]), dim=0)
            dcur, dhistory = group_fusion(dmerge, model, model_name, fusion_type, fout)
            for key in dcur:
                dinfos.setdefault(key, [])
                dinfos[key].append(dcur[key])
            if 'early_fusion' in dinfos and 'late_fusion' in dinfos:
                assert dinfos['early_trues'][-1].all() == dinfos['late_trues'][-1].all()
        aucs, accs = (dict(), dict())
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        aucs['concepts'] = auc
        accs['concepts'] = acc
        for key in dinfos:
            if key not in ['late_mean', 'late_vote', 'late_all', 'early_preds']:
                continue
            ts = np.concatenate(dinfos['late_trues'], axis=0)
            ps = np.concatenate(dinfos[key], axis=0)
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
            prelabels = [1 if p >= 0.5 else 0 for p in ps]
            acc = metrics.accuracy_score(ts, prelabels)
            aucs[key] = auc
            accs[key] = acc
    return (aucs, accs)
