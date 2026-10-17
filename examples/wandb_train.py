import os
import json
import torch
torch.set_num_threads(4)
from torch.optim import SGD, Adam
import copy
from pykt.models import train_model, init_model
from pykt.utils import debug_print, set_seed
from pykt.datasets import init_dataset4train
import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {'train_config': train_config, 'model_config': model_config, 'data_config': data_config, 'params': params}
    save_path = os.path.join(save_dir, 'config.json')
    with open(save_path, 'w') as fout:
        json.dump(d, fout)

def main(params):
    if 'use_wandb' not in params:
        params['use_wandb'] = 1
    if params['use_wandb'] == 1:
        import wandb
        wandb.init()
    set_seed(params['seed'])
    model_name, dataset_name, fold, emb_type, save_dir = (params['model_name'], params['dataset_name'], params['fold'], params['emb_type'], params['save_dir'])
    debug_print(text='load config files.', fuc_name='main')
    with open('../configs/kt_config.json') as f:
        config = json.load(f)
        train_config = config['train_config']
        if model_name in ['dkvmn', 'akt', 'robustkt']:
            train_config['batch_size'] = 64
        if model_name in ['simplekt', 'keenkt']:
            train_config['batch_size'] = 64
        if model_name == 'ukt':
            train_config['batch_size'] = 64
        model_config = copy.deepcopy(params)
        for key in ['model_name', 'dataset_name', 'emb_type', 'save_dir', 'fold', 'seed', 'batch_size', 'num_epochs']:
            model_config.pop(key, None)
        if 'batch_size' in params:
            train_config['batch_size'] = params['batch_size']
        if 'num_epochs' in params:
            train_config['num_epochs'] = params['num_epochs']
    batch_size, num_epochs, optimizer = (train_config['batch_size'], train_config['num_epochs'], train_config['optimizer'])
    with open('../configs/data_config.json') as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:
        train_config['seq_len'] = data_config[dataset_name]['maxlen']
    seq_len = train_config['seq_len']
    if model_name == 'ukt':
        model_config['use_question_difficulty'] = bool(params.get('use_question_difficulty', 1))
        model_config['edl_loss_type'] = params.get('edl_loss_type', 'mse')
    if model_name == 'ukt':
        model_config['use_CL'] = bool(params.get('use_CL', 0))
        model_config['use_uncertainty_aug'] = bool(params.get('use_uncertainty_aug', 1))
        model_config['use_edl'] = bool(params.get('use_edl', 0))
        model_config['edl_annealing_step'] = int(params.get('edl_annealing_step', 10))
        model_config['atten_type'] = params.get('atten_type', 'w2')
        print(f'{model_name} training mode:', 'EDL' if model_config['use_edl'] else 'BCE')
    if model_name == 'keenkt':
        model_config['use_CL'] = bool(params.get('use_CL', 0))
        model_config['use_uncertainty_aug'] = bool(params.get('use_uncertainty_aug', 1))
        model_config['use_diffusion'] = bool(params.get('use_diffusion', 0))
        model_config['diffusion_weight'] = float(params.get('diffusion_weight', 0.1))
        model_config['noise_level'] = float(params.get('noise_level', 0.2))
        model_config['use_edl'] = bool(params.get('use_edl', 1))
        model_config['edl_annealing_step'] = int(params.get('edl_annealing_step', 10))
        model_config['num_classes'] = int(params.get('num_classes', 2))
        model_config['atten_type'] = params.get('atten_type', 'w2')
        print('keenkt training mode:', 'EDL' if model_config['use_edl'] else 'BCE', f"use_CL={model_config['use_CL']}", f"use_diffusion={model_config['use_diffusion']}")
    print('Start init data')
    print(dataset_name, model_name, data_config, fold, batch_size)
    debug_print(text='init_dataset', fuc_name='main')
    train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
    params_str = '_'.join([str(v) for k, v in params.items() if not k in ['other_config']])
    print(f'params: {params}, params_str: {params_str}')
    if params['add_uuid'] == 1 and params['use_wandb'] == 1:
        import uuid
        params_str = params_str + f'_{str(uuid.uuid4())}'
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f'Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}')
    print(f'model_config: {model_config}')
    print(f'train_config: {train_config}')
    if model_name in ['atdkt', 'simplekt', 'keenkt', 'ukt']:
        model_config['seq_len'] = seq_len
    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params['learning_rate']
    for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
        if remove_item != 'l2' or model_name != 'atdkt':
            if remove_item in model_config:
                del model_config[remove_item]
    debug_print(text='init_model', fuc_name='main')
    print(f'model_name:{model_name}')
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f'model is {model}')
    if model_name in ['ukt', 'keenkt']:
        opt = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-05)
    elif optimizer == 'sgd':
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        opt = Adam(model.parameters(), learning_rate)
    validauc, validacc = (-1, -1)
    best_epoch = -1
    save_model = True
    debug_print(text='train model', fuc_name='main')
    _testauc, _testacc, _window_testauc, _window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model)
    if save_model:
        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        net = torch.load(os.path.join(ckpt_path, emb_type + '_model.ckpt'))
        best_model.load_state_dict(net)
    print('fold\tmodelname\tembtype\tvalidauc\tvalidacc\tbest_epoch')
    print(f'{fold}\t{model_name}\t{emb_type}\t{validauc}\t{validacc}\t{best_epoch}')
    model_save_path = os.path.join(ckpt_path, emb_type + '_model.ckpt')
    print(f'end:{datetime.datetime.now()}')
    if params['use_wandb'] == 1:
        wandb.log({'validauc': validauc, 'validacc': validacc, 'best_epoch': best_epoch, 'model_save_path': model_save_path})
