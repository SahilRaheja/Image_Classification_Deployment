import numpy as np 
import pandas as pd
import argparse

from src.utils.common import read_config
from src.utils.data import load_data
from src.utils.model import load_model

def training(config_path):
    config = read_config(config_path)

    train_path = config["params"]["train_path"]
    test_path = config["params"]["test_path"]
    shear_range = config["params"]["shear_range"]
    zoom_range = config["params"]["zoom_range"]
    horizontal_flip = config["params"]["horizontal_flip"]
    target_size = config["params"]["target_size"]
    batch_size = config["params"]["batch_size"]
    

    training_set, test_set = load_data(train_path, test_path, shear_range, zoom_range, 
                            horizontal_flip, target_size, batch_size)

    loss_function = config["params"]["loss_function"]
    optimizer = config["params"]["optimizer"]
    metrics = config["params"]["metrics"]
    
    classifier = load_model(loss_function, optimizer, metrics)
    
    steps_per_epoch = config["params"]["steps_per_epoch"]
    epochs = config["params"]["epochs"]
    validation_steps = config["params"]["validation_steps"]   

    classifier.fit(training_set, steps_per_epoch=steps_per_epoch,epochs=epochs, validation_data=test_set, validation_steps=validation_steps) 
    classifier.save('artifacts/model.h5')

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)

    