import errno
import os, json
from pathlib import Path
from shutil import rmtree
from dotmap import DotMap

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def save(file, text):
    """ Saves a text in a text file. """
    with open(file, 'w') as fp:
        fp.write(str(text))

def load_json(filename):
    """ loads a json file into a list (od dictionaries) """
    with open(filename,'r') as json_file:
        data = json.load(json_file)
    return data

def load_yaml(filename):
    with open(filename,'r') as json_file:
        data = yaml.load(json_file, Loader=Loader)
    return data

def save_json(data, filename):
    """ Saves data as json. """
    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4)

def write(file, string):
    with open(file, "w") as fp:
        fp.write(string)


def read_config(filename):
    """ Read a dictionary in a configuration format and transforms it into DotMap."""
    if filename.endswith(".json"):
        data = load_json(filename)
    elif filename.endswith(".yaml"):
        data = load_yaml(filename)
        
    return DotMap(data)

def create_dir(path, clear=False):
    """ Creates a directory on disk. """
    if clear and os.path.exists(path):
        rmtree(path)
    
    try:
        Path(path).mkdir(parents=True, exist_ok=not clear) 
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise