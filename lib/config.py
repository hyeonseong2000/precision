import yaml
import argparse

def convert_to_namespace (dict_in, args=None, dont_care=None):
    if args is None:
        args = argparse.Namespace()

    for ckey, cvalue in dict_in.items():
        #if ckey not in args.__dict__.keys():
        #if (dont_care is not None) and ckey in dont_care.keys():
        if (dont_care is not None) and (ckey in dont_care):
            pass # keep value in original name space
        else:
            args.__dict__[ckey] = cvalue
      
    return args


def parse_config_yaml_dict (yaml_path, args=None):
    with open (yaml_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    #print (configs)
    if configs is not None:
        base_config = configs.get('base_config')

        if base_config is not None:
            base_config = parse_config_yaml_dict(configs['base_config'])

            if base_config is not None:
                # update contents in base configs w/ config
                configs = update_recursive(src_dict=configs, dst_dict=base_config) 
                print(configs)
            else:
                raise FileNotFoundError('base_config specified but not found')

    return configs


def update_recursive(src_dict, dst_dict):
    for k, v in src_dict.items():
        if k not in dst_dict:
            dst_dict[k] = dict()
        if isinstance(v, dict):
            update_recursive(src_dict=v, dst_dict=dst_dict[k])
        else:
            dst_dict[k] = v
    return dst_dict



def parse_config_yaml_namespace (yaml_path, args=None):
    with open(yaml_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    return configs




