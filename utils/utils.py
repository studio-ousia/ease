import pickle
import numpy as np
import torch
import os


def pickle_dump(obj, path):
    with open(path, mode="wb") as f:
        print(f"save to {path}")
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, mode="rb") as f:
        print(f"load from {path}")
        data = pickle.load(f)
        return data


def save_model(output_dir, model, info):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), f"{output_dir}/model.bin")
    torch.save(info, f"{output_dir}/info.bin")


def create_numpy_sequence(source_sequence, length, dtype):
    ret = np.zeros(length, dtype=dtype)
    source_sequence = source_sequence[:length]
    ret[: len(source_sequence)] = source_sequence
    return ret


def update_args(base_args, input_args):
    for key, value in dict(input_args).items():
        base_args.__dict__[key] = value
    return base_args
