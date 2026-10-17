from pykt.config import SUPPORTED_MODELS
import torch
import os
from .dkt import DKT
from .dkvmn import DKVMN
from .akt import AKT
from .atdkt import ATDKT
from .simplekt import simpleKT
from .keenkt import KEENKT
from .ukt import UKT
from .robustkt import Robustkt
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

def init_model(model_name, model_config, data_config, emb_type):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported model: {model_name}; supported models: {SUPPORTED_MODELS}')
    if model_name == 'dkt':
        model = DKT(data_config['num_c'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    elif model_name == 'dkvmn':
        model = DKVMN(data_config['num_c'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    elif model_name == 'akt':
        model = AKT(data_config['num_c'], data_config['num_q'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    elif model_name == 'atdkt':
        model = ATDKT(data_config['num_q'], data_config['num_c'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    elif model_name == 'simplekt':
        model = simpleKT(data_config['num_c'], data_config['num_q'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    elif model_name == 'keenkt':
        model = KEENKT(data_config['num_c'], data_config['num_q'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    elif model_name == 'ukt':
        model = UKT(data_config['num_c'], data_config['num_q'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    elif model_name == 'robustkt':
        model = Robustkt(data_config['num_c'], data_config['num_q'], **model_config, emb_type=emb_type, emb_path=data_config['emb_path']).to(device)
    else:
        print('The wrong model name was used...')
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type + '_model.ckpt'))
    model.load_state_dict(net)
    return model
