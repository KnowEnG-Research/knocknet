from argparse import ArgumentParser
import os
import yaml
import params
from models import Model


def train(param_dict=params.default_param_dict):
    model = Model(param_dict, queue=queue)


def main():
    parser = ArgumentParser()
    parser = params.add_trainer_args(parser)
    param_dict = vars(parser.parse_args())
    param_yml = os.path.join(param_dict["log_dir"], 'params.yml')
    with open(param_yml, 'w') as outfile:
        yaml.dump(param_dict, outfile, default_flow_style=False)
    train(param_dict)


if __name__ == '__main__':
    main()
