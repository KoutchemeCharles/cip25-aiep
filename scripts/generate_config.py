""" Class for generating all the experiments that 
are going to be ran for the paper. 

"""

import os 
import glob 

from argparse import ArgumentParser
from src.utils.files import (
    create_dir, load_json, load_yaml, save_json
)


def parse_args():
    description = "Generating experiments configurations"
    parser = ArgumentParser(description=description)
    parser.add_argument("--name", required=True, 
                        help="Name of the experiments.")
    parser.add_argument("--model", required=True,
                        help="Which model to use")
    parser.add_argument("--dataset", required=True, 
                        nargs="+", help="One or more dataset paths")
    parser.add_argument("--task", required=True,
                        help="Task to run")
    parser.add_argument("--config_dir",
                        help="Path towards the folder where to save the experiment configs.",
)
    parser.add_argument("--save_dir",
                        help="Path towards the folder where the experiments results are going to be saved")
    return parser.parse_args()


def main():
    args = vars(parse_args())
    create_experiment(args)


def create_experiment(args):
    config_name = _create_config_name(args)
    config = {
        "name": config_name,
        "save_dir": os.path.join(args["save_dir"], args["name"]),
        "seed": 42
    }

    other_keys = [k for k in args if k not in config]

    for k in other_keys:
        if k == "dataset":
            dataset_files = []
            for ds_path in args[k]:
                matches = sorted(glob.glob(ds_path))
                if not matches:
                    raise ValueError(f"No files matched pattern: {ds_path}")
                dataset_files.extend(matches)
            config[k] = [load(ds_file) for ds_file in dataset_files]
        else:
            config[k] = load(args[k]) if isinstance(args[k], str) and (args[k].endswith(".json") or args[k].endswith(".yaml")) else args[k]


    configs_save_dir = os.path.join(args["config_dir"], args["name"])
    create_dir(configs_save_dir)
    config_id = len(os.listdir(configs_save_dir))
    save_path = os.path.join(configs_save_dir, str(config_id)  + ".json")
    save_json(config, save_path)
    

def load(path):
    if isinstance(path, str):  # Already loaded
        return _load(path)
    elif isinstance(path, list):
        return list(map(_load, path))
    else:
        raise ValueError()
    
def _load(path):
    if path.endswith(".json"):
        return load_json(path)
    elif path.endswith(".yaml"):
        return load_yaml(path)
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def _create_config_name(params):
    m = params["model"].split("/")[-1].replace(".yaml", "")
    t = params["task"].split("/")[-1].replace(".yaml", "")
    # Now params["dataset"] is a list!
    if isinstance(params["dataset"], list):
        d = "+".join([os.path.basename(ds).replace(".yaml", "") for ds in params["dataset"]])
    else:
        d = params["dataset"].split("/")[-1].replace(".yaml", "")
    return "{m}_{t}_{d}".format(m=m, t=t, d=d)



if __name__ == "__main__":
    main()


# TODO: BONUS - extend later to take multiple arguments which allow ParameterConfig