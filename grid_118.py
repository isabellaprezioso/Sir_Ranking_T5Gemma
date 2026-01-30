# -*- encoding: utf-8 -*-
import fnmatch
import os  # os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
import models
from dataset import ReviewData
from framework import Model
import json


def grid_filename(model, new_grid=True):
    from main import now

    def prefix_search(folder, prefix):
        # Elenca tutti i file nella cartella
        temp_file = os.listdir(folder)

        # Filtra i file che iniziano con il prefisso specificato
        temp_file = fnmatch.filter(temp_file, f"{prefix}*")

        return temp_file

    grid = model + "_"

    files = prefix_search(folder="./grid/", prefix=grid)

    if new_grid:
        files = []

    if len(files) == 0:
        temp = grid + now() + ".txt"
        files = [temp.replace('-', '_').replace(' ', '!').replace(':', '_')]

    files.sort(reverse=True)
    files = files[0]
    files = files[:len(files) - 4]

    return files


# Method to resume grid search if anything happened
def resume_grid(grid):
    # Check if log file exists, if not it will be created
    if not os.path.exists(grid):
        with open(grid, 'w') as file:
            json.dump([], file)

    with open(grid, 'r') as file:
        data = json.load(file)

    # "Combinations" is an array of all the combinations of hyperparameters already processed
    # Which means EXCLUDE all these combinations from the grid search
    combinations = []

    # Interrupted grid info
    best_loss = float("inf")

    resume = {}

    for item in data:

        if best_loss > item["best_loss"]:
            best_loss = item["best_loss"]
            resume["best_hyperparameters"] = item["hyperparameters"]

        # Already processed hyperparameters -> Exclude
        combinations.append(item["hyperparameters"])

    resume["combinations"] = combinations
    resume["best_loss"] = best_loss

    return resume


def resume_training(resume):
    best_combination = None
    # set the best hyperparameters again
    if 'best_hyperparameters' in resume:
        best_combination = resume["best_hyperparameters"]
        del resume["best_hyperparameters"]

    best_result = float('inf')
    # set the best loss again
    if 'best_loss' in resume:
        best_result = resume["best_loss"]
        del resume["best_loss"]

    return best_result, best_combination


def update_json(data, grid):
    # Check if log file exists, if not it will be created
    if not os.path.exists(grid):
        with open(grid, 'w') as file:
            json.dump([], file)

    with open(grid, 'r') as file:
        info = json.load(file)

    info.append(data)
    with open(grid, 'w') as file:
        json.dump(info, file)
