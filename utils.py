import pickle
import json

def load_data(saved_file_dir="../saved_file"):
    with open('{}/X_train.pkl'.format(saved_file_dir), 'rb') as f:
        train_x = pickle.load(f)
    with open('{}/y_train.pkl'.format(saved_file_dir), 'rb') as f:
        train_y = pickle.load(f)
    with open('{}/X_valid.pkl'.format(saved_file_dir), 'rb') as f:
        valid_x = pickle.load(f)
    with open('{}/y_valid.pkl'.format(saved_file_dir), 'rb') as f:
        valid_y = pickle.load(f)
    return train_x, train_y, valid_x, valid_y


def load_param(trial):
    with open('params.json', 'rb') as f:
        js = json.load(f)
    return js[trial]