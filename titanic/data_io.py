import csv
import json
import os
import pandas as pd
import pickle

def get_paths():
    paths = json.loads(open("Settings.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def get_train_df():
    train_path = get_paths()["train_data_path"]
    return pd.read_csv(train_path)

def get_test_df():
    valid_path = get_paths()["test_data_path"]
    return pd.read_csv(valid_path)

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def write_submission(predictions, rowIds, columnsNameList):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    rows = [x for x in zip(rowIds, predictions.flatten())]
    writer.writerow((columnsNameList[0], columnsNameList[1]))
    writer.writerows(rows)