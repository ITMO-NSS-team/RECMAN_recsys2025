import argparse
from . import defaults


class ArgparserException(Exception): pass


def parse_args(is_test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--time_offset', default=defaults.time_offset_q, type=float)
    parser.add_argument('--target_metric', choices=('HR', 'MRR', 'NDCG'), default=defaults.target_metric, type=str)
    parser.add_argument('--topn', default=10, type=int)
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('--exhaustive', default=False, action="store_true")
    parser.add_argument('--grid_steps', default=None, type=int) # 0 means run infinitely, None will use defaults
    parser.add_argument('--check_best', default=False, action="store_true")
    parser.add_argument('--save_config', default=False, action="store_true")
    parser.add_argument('--dump_results', default=False, action="store_true")
    parser.add_argument('--es_tol', default=0.001, type=float)
    parser.add_argument('--es_max_steps', default=2, type=int)
    parser.add_argument('--next_item_only', default=False, action="store_true")
    # saving/resuming studies via RDB:
    parser.add_argument('--study_name', default=None, type=str)
    parser.add_argument('--storage', choices=('sqlite', 'redis'), default=None, type=str)
    #args = parser.parse_args(['--model','sasrec','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/sasrec.py', '--grid_steps', '60','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/hypsasrec_ML1M_SVD_e-5.py', '--grid_steps', '60','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/sasrec_manifold.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb_manifold','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/sasrecb_manifold.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/best/sasrec_ml1m.py','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/best/sasrec_ml1m.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/best/sasrec_ml1m.py','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrec','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/best/sasrec_ml1m.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/best/sasrec_ml1m.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb_manifold','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/best/sasrec_ml1m.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','ml-1m','--time_offset','0.95','--config_path','./grids/best/sasrec_ml1m.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/sasrec.py', '--grid_steps', '60','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/hypsasrec.py', '--grid_steps', '60','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/sasrec.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/sasrec_manifold.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/best/sasrec_Digital_Music.py','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/best/sasrec_Digital_Music.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/best/sasrec_Digital_Music.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb','--dataset','Digital_Music','--time_offset','0.95','--config_path','./grids/best/sasrec_Digital_Music.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','Digital_Music_5','--time_offset','0.95','--config_path','./grids/best/sasrec_dig.py','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','Digital_Music_5','--time_offset','0.95','--config_path','./grids/best/sasrec_dig.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Digital_Music_5','--time_offset','0.95','--config_path','./grids/sasrec_manifold.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Digital_Music_5','--time_offset','0.95','--config_path','./grids/best/sasrec_dig.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb','--dataset','Digital_Music_5','--time_offset','0.95','--config_path','./grids/best/sasrec_dig.py','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrec','--dataset','Digital_Music_5','--time_offset','0.95','--config_path','./grids/best/sasrec_dig.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb_manifold','--dataset','Digital_Music_5','--time_offset','0.95','--config_path','./grids/best/sasrec_dig.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Arts_Crafts_and_Sewing_5','--time_offset','0.95','--config_path','./grids/sasrec_manifold.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','Arts_Crafts_and_Sewing_5','--time_offset','0.95','--config_path','./grids/best/sasrec_arts.py','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','Arts_Crafts_and_Sewing_5','--time_offset','0.95','--config_path','./grids/best/sasrec_arts.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Arts_Crafts_and_Sewing_5','--time_offset','0.95','--config_path','./grids/best/sasrec_arts.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb','--dataset','Arts_Crafts_and_Sewing_5','--time_offset','0.95','--config_path','./grids/best/sasrec_arts.py','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrec','--dataset','Arts_Crafts_and_Sewing_5','--time_offset','0.95','--config_path','./grids/best/sasrec_arts.py','--dump_results'])
    args = parser.parse_args(['--model','sasrecb_manifold','--dataset','Arts_Crafts_and_Sewing_5','--time_offset','0.95','--config_path','./grids/best/sasrec_arts.py','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Luxury_Beauty_5','--time_offset','0.95','--config_path','./grids/sasrec_manifold.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','Luxury_Beauty_5','--time_offset','0.95','--config_path','./grids/best/sasrec_lux.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','Luxury_Beauty_5','--time_offset','0.95','--config_path','./grids/best/sasrec_lux.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Luxury_Beauty_5','--time_offset','0.95','--config_path','./grids/best/sasrec_lux.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb','--dataset','Luxury_Beauty_5','--time_offset','0.95','--config_path','./grids/best/sasrec_lux.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrec','--dataset','Luxury_Beauty_5','--time_offset','0.95','--config_path','./grids/best/sasrec_lux.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb_manifold','--dataset','Luxury_Beauty_5','--time_offset','0.95','--config_path','./grids/best/sasrec_lux.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Office_Products_5','--time_offset','0.95','--config_path','./grids/sasrec_manifold.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec','--dataset','Office_Products_5','--time_offset','0.95','--config_path','./grids/best/sasrec_office.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrecb','--dataset','Office_Products_5','--time_offset','0.95','--config_path','./grids/best/sasrec_office.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrec_manifold','--dataset','Office_Products_5','--time_offset','0.95','--config_path','./grids/best/sasrec_office.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','sasrecb','--dataset','Office_Products_5','--time_offset','0.95','--config_path','./grids/best/sasrec_office.py', '--grid_steps', '0','--dump_results'])
    #args = parser.parse_args(['--model','hypsasrec','--dataset','Office_Products_5','--time_offset','0.95','--config_path','./grids/best/sasrec_office.py', '--grid_steps', '0','--dump_results'])
    validate_args(args, is_test)
    return args


def validate_args(args, is_test):
    if not is_test and not args.config_path:
        # models that require hyper-params tuning must be provided with config file
        # models without hyper-parameters are only valid for test not for tuning phase
        raise ArgparserException('`config_path` must be provided for tuning.')
