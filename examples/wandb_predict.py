import os
import argparse
import json
import copy
import torch
import pandas as pd
from pykt.models import evaluate, evaluate_edl, evaluate_question, analyze_edl_uncertainty, load_model
from pykt.datasets import init_test_datasets
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
with open('../configs/wandb.json') as fin:
    wandb_config = json.load(fin)

def main(params):
    if params['use_wandb'] == 1:
        import wandb
        os.environ['WANDB_API_KEY'] = wandb_config['api_key']
        wandb.init(project='wandb_predict')
    save_dir = params['save_dir']
    batch_size = params['bz']
    fusion_type = params['fusion_type'].split(',')
    with open(os.path.join(save_dir, 'config.json')) as fin:
        config = json.load(fin)
    model_config = copy.deepcopy(config['model_config'])
    trained_model_name = config.get('params', {}).get('model_name', '')
    remove_items = ['use_wandb', 'learning_rate', 'add_uuid']
    if trained_model_name != 'atdkt':
        remove_items.append('l2')
    for remove_item in remove_items:
        model_config.pop(remove_item, None)
    trained_params = config['params']
    fold = trained_params['fold']
    model_name = trained_params['model_name']
    dataset_name = trained_params['dataset_name']
    emb_type = trained_params['emb_type']
    if model_name in ['atdkt']:
        train_config = config['train_config']
        model_config['seq_len'] = train_config['seq_len']
    with open('../configs/data_config.json') as fin:
        curconfig = copy.deepcopy(json.load(fin))
    data_config = curconfig[dataset_name]
    data_config['dataset_name'] = dataset_name
    if int(params.get('skip_question_eval', 0)) == 1:
        data_config = copy.deepcopy(data_config)
        data_config.pop('test_question_file', None)
        data_config.pop('test_question_window_file', None)
        print('Skip loading question-level test datasets.')
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    print(f'Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}')
    print(f'model_config: {model_config}')
    print(f'data_config: {data_config}')
    model = load_model(model_name, model_config, data_config, emb_type, save_dir)
    is_edl_model = model.model_name in ['ukt', 'akt', 'dkt', 'robustkt', 'dkvmn', 'atdkt', 'simplekt', 'keenkt'] and getattr(model, 'use_edl', False)
    rel = None
    save_test_path = os.path.join(save_dir, model.emb_type + '_test_predictions.txt')
    uncertainty_result = None
    window_uncertainty_result = None
    if is_edl_model:
        edl_analysis_dir = os.path.join(save_dir, 'edl_analysis')
        testauc, testacc, test_edl_df = evaluate_edl(model=model, test_loader=test_loader)
        uncertainty_result = analyze_edl_uncertainty(result_df=test_edl_df, output_dir=edl_analysis_dir, prefix='test')
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, save_path=save_test_path)
    print(f'testauc: {testauc}, testacc: {testacc}')
    window_testauc = -1
    window_testacc = -1
    save_test_window_path = os.path.join(save_dir, model.emb_type + '_test_window_predictions.txt')
    if is_edl_model:
        window_testauc, window_testacc, window_edl_df = evaluate_edl(model=model, test_loader=test_window_loader)
        window_uncertainty_result = analyze_edl_uncertainty(result_df=window_edl_df, output_dir=edl_analysis_dir, prefix='test_window')
    else:
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_path=save_test_window_path)
    print(f'testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}')
    dres = {'testauc': testauc, 'testacc': testacc, 'window_testauc': window_testauc, 'window_testacc': window_testacc}
    if is_edl_model and uncertainty_result is not None:
        summary = uncertainty_result['summary']
        dres.update({'mud': summary['mud'], 'w2': summary['w2']})
    if is_edl_model and window_uncertainty_result is not None:
        summary = window_uncertainty_result['summary']
        dres.update({'window_mud': summary['mud'], 'window_w2': summary['w2']})
    q_testaucs = -1
    q_testaccs = -1
    qw_testaucs = -1
    qw_testaccs = -1
    run_question_evaluation = not is_edl_model and params['skip_question_eval'] == 0
    if run_question_evaluation and 'test_question_file' in data_config and (test_question_loader is not None):
        save_test_question_path = os.path.join(save_dir, model.emb_type + '_test_question_predictions.txt')
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres['oriauc' + key] = q_testaucs[key]
        for key in q_testaccs:
            dres['oriacc' + key] = q_testaccs[key]
    if run_question_evaluation and 'test_question_window_file' in data_config and (test_question_window_loader is not None):
        save_test_question_window_path = os.path.join(save_dir, model.emb_type + '_test_question_window_predictions.txt')
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres['windowauc' + key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres['windowacc' + key] = qw_testaccs[key]
    print(dres)
    print('\n========== Final reported metric ==========')
    if is_edl_model:
        print(f'{model.model_name.upper()}+EDL knowledge-point evaluation:')
        print(f'  testauc = {testauc:.6f}')
        print(f'  testacc = {testacc:.6f}')
        print(f'  window_testauc = {window_testauc:.6f}')
        print(f'  window_testacc = {window_testacc:.6f}')
        if uncertainty_result is not None:
            print(f"  MUD = {dres['mud']:.6f}")
            print(f"  W2 = {dres['w2']:.6f}")
        if window_uncertainty_result is not None:
            print(f"  window_MUD = {dres['window_mud']:.6f}")
            print(f"  window_W2 = {dres['window_w2']:.6f}")
        print('  Question-level fusion was skipped.')
    elif 'windowauclate_mean' in dres and 'windowacclate_mean' in dres:
        print('Question-level windowed test metric:')
        print(f"  windowauclate_mean = {dres['windowauclate_mean']:.6f}")
        print(f"  windowacclate_mean = {dres['windowacclate_mean']:.6f}")
    else:
        print('Concept-level windowed test metric:')
        print(f'  window_testauc = {window_testauc:.6f}')
        print(f'  window_testacc = {window_testacc:.6f}')
    print('===========================================\n')
    raw_config = json.load(open(os.path.join(save_dir, 'config.json')))
    dres.update(raw_config['params'])
    final_metrics_path = os.path.join(save_dir, model.emb_type + '_final_test_metrics.csv')
    pd.DataFrame([dres]).to_csv(final_metrics_path, index=False, encoding='utf-8-sig')
    print(f'Final test metrics saved to: {final_metrics_path}')
    if params['use_wandb'] == 1:
        wandb.log(dres)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bz', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='saved_model')
    parser.add_argument('--fusion_type', type=str, default='early_fusion,late_fusion')
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--skip_question_eval', type=int, default=0)
    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
