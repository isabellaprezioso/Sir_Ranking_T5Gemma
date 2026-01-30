# -*- encoding: utf-8 -*-
import itertools
import json
import time
import random
import math
import fire
import pandas as pd
import shutil
import os

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ReviewData
from framework import Model
from models.Losses import *
import models
import config

from grid_118 import grid_filename, resume_grid, resume_training, update_json

def core_unpack(opt, uids, iids):
    uids = list(uids)
    iids = list(iids)

    user_reviews = opt.users_review_list[uids]
    user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id

    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]  # 检索出该item对应的user id
    item_doc = opt.item_doc[iids]
    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    data = list(map(lambda x: torch.LongTensor(x), data))
    return data


def unpack_input(opt, x, setup="Default"):
    if setup == "Default":
        uids, iids = list(zip(*x))
        data = core_unpack(opt, uids, iids)
        return data

    if setup == "BPR":
        uids, pos_iids, neg_iids = list(zip(*x))
        pos_data = core_unpack(opt, uids, pos_iids)
        neg_data = core_unpack(opt, uids, neg_iids)
        return pos_data, neg_data

    return None


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(setup="Default"):
    def collate_fn_default(batch):
        data, label = zip(*batch)
        return data, label

    def collate_fn_bpr(batch):
        user, pos_item, neg_item = zip(*batch)
        return user, pos_item, neg_item

    if setup == "Default":
        return collate_fn_default
    if setup == "BPR":
        return collate_fn_bpr

    return None


def train(**kwargs):
    grid = False
    if "grid" in kwargs:
        grid = kwargs["grid"]
        kwargs.pop("grid", None)

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn())

    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn())

    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)

            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss

            loss.backward()

            optimizer.step()
            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
                    if val_loss < min_loss:
                        model.save(name=opt.dataset, opt=opt.print_opt)
                        min_loss = val_loss
                        print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss

        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};")

        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        if val_loss < min_loss:
            model.save(name=opt.dataset, opt=opt.print_opt)
            print(opt.dataset + " | " + opt.print_opt)
            min_loss = val_loss
            print("model save")
        if val_mse < best_res:
            best_res = val_mse
        print("*" * 30)

    print("----" * 20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    print("----" * 20)

    if grid:
        return best_res, opt.dataset, opt.print_opt


def test(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    assert (len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test dataset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)

            output = model(test_data)
            mse_loss = torch.sum((output - scores) ** 2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output - scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    print(f"\tevaluation result: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")
    model.train()
    return total_loss, mse, mae


def using(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()

    user = 0
    if 'user' in kwargs:
        user = kwargs['user']
        kwargs.pop('user')

    opt.parse(kwargs)

    assert (len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Inference", user=user)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn())
    print(f"{now()}: test in the test dataset")
    df = predict_df(model, test_data_loader, opt)
    return df


def predict_df(model, data_loader, opt):
    predictions = []  # Lista per le previsioni
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            # Visualizza il contenuto di test_data
            user_id = []  # Inizializza lista vuota per user_id
            item_id = []  # Inizializza lista vuota per item_id

            # Aggiungi gli elementi alle liste
            for sample in test_data:
                user_id.append(sample[0])
                item_id.append(sample[1])

            test_data = unpack_input(opt, test_data)
            output = model(test_data)
            # Aggiungi le previsioni alla lista
            for i in range(len(output)):
                predictions.append((user_id[i], item_id[i], output[i].item()))
    df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'output'])
    df.drop_duplicates(ignore_index=True)
    print(df)
    model.train()
    return df


def top_predictions_for_user(df, k=10):
    # Ordina le previsioni in base all'output decrescente
    user_df_sorted = df.sort_values(by='output', ascending=False)

    # Restituisce le prime k tuple
    top_k_predictions = user_df_sorted.head(k)
    print('Top predictions = ')
    print(top_k_predictions)

    # return top_k_predictions


def combine(**kwargs):
    df = using(**kwargs)
    top_predictions_for_user(df)


def grid_hyperparameters():
    hyperparameters = {
        "lr": [1e-3, 2e-3, 1e-2],  # Learning Rate
        "kernel_size": [3, 4, 5],  # "t" window size
        "word_dim": [200, 300, 500],  # "c" dimension word embedding
        # dropout       = 0.5
        # weight_decay  = 1e-3
        # optimizer     = AdamW
        # epochs        = 100
        # batch_size    = 128
    }

    return hyperparameters


def swap_best_model(dataset, print_opt, model):
    # Saving best.pth weights from GridSearch
    model = model + "_" + dataset + "_" + print_opt  # Take model name

    pth_from = "./checkpoints/" + model + ".pth"  # Path to .pth file > to copy
    pth_best = "./checkpoints/" + model + "_GridBEST.pth"  # Path to new .pth  > to save

    shutil.copy2(pth_from, pth_best)  # copy pth_from into pth_best


def grid_search(**kwargs):

    model = "DeepCoNN"
    if "model" in kwargs:
        model = kwargs["model"]

    setup = False
    if "setup" in kwargs:
        setup = kwargs["setup"]
        del kwargs["setup"]

    kwargs['print_opt'] = setup
    datas = False
    if "dataset" in kwargs:
        datas = kwargs["dataset"]

    os.makedirs("./grid/" + datas, exist_ok=True)

    tmp = model + "_" + setup
    grid_info = grid_filename(model=tmp, new_grid=False, datas=datas)
    save_name = grid_info
    grid_info = "./grid/" + datas + "/" + grid_info

    grid_txt = grid_info + ".txt"
    grid_json = grid_info + ".json"

    # Saving starting time of the grid search

    with open(grid_txt, 'a') as file:
        file.write("Grid search started at " + str(now()) + "\n\n")

    # Adding new element to kwargs (useful information for train())
    if "grid" not in kwargs:
        kwargs["grid"] = True

    # Generating a log.txt file related to the grid search
    if not os.path.exists(grid_txt):
        with open(grid_txt, 'w') as file:
            pass

    # Creating all combinations given the hyperparameters
    keys, values = zip(*grid_hyperparameters().items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Saving best set of hyperparameters with related loss
    best_combination = None
    best_result = float('inf')

    resume = resume_grid(grid=grid_json)

    # grid search interrupted midway
    if resume:
        # filter processed combinations
        if 'combinations' in resume:
            combinations = [con for con in combinations if con not in resume['combinations']]
            del resume['combinations']

        best_result, best_combination = resume_training(resume)

    for combination in combinations:

        # Saving starting time for one training
        start = time.time()

        # Training with a specific combination of hyperparameters
        print(f"Training with hyperparameters: {combination}")

        temp = "Hyperparameters: " + str(combination) + "\n\n"

        search = ""
        for _ in range(0, len(temp)):
            search += "#"
        search += "\n" + temp
        search += "Started at " + str(now()) + "\n\n"

        with open(grid_txt, 'a') as file:
            file.write(search)

        if setup == "Default":
            result, dataset, print_opt = train(**combination, **kwargs)
        elif setup == "BPR":
            from BPR import train_bpr
            result, dataset, print_opt = train_bpr(**combination, **kwargs)
        else:
            raise Exception("No valid ranking train has been chosen")

        # Saving ending time for one training
        end = time.time()

        # Evaluating training time with given hyperparameters
        hours, reminder = divmod(end - start, 3600)
        minutes, seconds = divmod(reminder, 60)

        (hours, minutes, seconds) = (int(hours), int(minutes), int(seconds))
        if hours < 10:
            hours = "0" + str(int(hours))  # 00h
        if minutes < 10:
            minutes = "0" + str(int(minutes))  # 00m
        if seconds < 10:
            seconds = "0" + str(int(seconds))  # 00s

        # Saving information about a specific computed combination
        search = "Validation loss: " + str(result) + "\n"
        search += "Computed time: [" + str(hours) + "h:" + str(minutes) + "m] (" + str(seconds) + "s)\n\n"

        with open(grid_txt, 'a') as file:
            file.write(search)

        if result < best_result:
            swap_best_model(dataset=dataset, print_opt=print_opt, model=kwargs['model'])

            # Save best results and combination of hyperparameters
            best_result = result
            best_combination = combination

        data = {
            "best_loss": result,
            "computed_time": int(end - start),
            "hyperparameters": combination
        }

        update_json(data, grid_json)

    with open(grid_json, 'r') as file:
        info = json.load(file)
        computed_time = 0
        for item in info:
            computed_time += item['computed_time']

    # Evaluating grid search computed time
    hours, reminder = divmod(computed_time, 3600)
    minutes, seconds = divmod(reminder, 60)

    (hours, minutes, seconds) = (int(hours), int(minutes), int(seconds))
    if hours < 10:
        hours = "0" + str(int(hours))  # 00h
    if minutes < 10:
        minutes = "0" + str(int(minutes))  # 00m
    if seconds < 10:
        seconds = "0" + str(int(seconds))  # 00s

    ending_info = "Grid search ended at " + str(now()) + "\n"
    ending_info += "Best hyperparameters: " + str(best_combination) + "\n"
    ending_info += "Best validation loss: " + str(best_result) + "\n"
    ending_info += "Computed time: [" + str(hours) + "h:" + str(minutes) + "m] (" + str(seconds) + "s)\n\n"

    with open(grid_txt, 'a') as file:
        file.write(ending_info)

    finished = "./grid/"+datas+"finished/"
    os.makedirs(finished, exist_ok=True)

    shutil.move(grid_txt, finished + save_name + ".txt")
    shutil.move(grid_json, finished + save_name + ".json")

    print(f"Best hyperparameters: {best_combination}")
    print(f"Best result: {best_result}")


def args(type=None):
    kwargs = False
    if type == "DeepCoNN_train":
        kwargs = {
            'model': 'DeepCoNN',
            'num_fea': 1,
            'output': 'fm',
            'use_gpu': True,
            'setup': 'BPR',
            'dataset': 'Toys_and_Games_data'
        }

    if type == "NARRE_train_BPR":
        kwargs = {
            'model': 'NARRE',
            'num_fea': 2,
            'output': 'lfm',
            'use_gpu': True,
            'setup': 'BPR'
        }

    if type == "DeepCoNN_test_BPR":
        kwargs = {
            'model': 'DeepCoNN',
            'num_fea': 1,
            'output': 'fm',
            'use_gpu': True,
            'pth_path': './checkpoints/DeepCoNN_Digital_Music_data_default_GridBEST.pth',
        }

    if type == "NARRE_train":
        kwargs = {
            'model': 'NARRE',
            'num_fea': 2,
            'output': 'lfm',
            'use_gpu': True,
            'word_dim': 300
        }

    if type == "using":
        kwargs = {
            'dataset': 'Digital_Music_data',
            'pth_path': './checkpoints/DeepCoNN_Digital_Music_data_default_GridBEST.pth',
            'model': 'DeepCoNN',
            'user': 129
        }

    return kwargs


if __name__ == "__main__":
    fire.Fire()
    # foo = args("train")
    # train(**args("DeepCoNN_train"))
    # train_bpr(**args("DeepCoNN_train"))
    # grid_search(**args('DeepCoNN_train'))
    # start = int(time.time())
    # grid_filename(model="NARRE")
    # resume_grid(grid="DeepCoNN_2024_07_11___15_14_15.txt",)

    # Chiamata alla funzione using con gli argomenti specificati
    # DF = using(**args("using"))
    # DF.sort_values(by='output', inplace=True, ascending=False)
    # top = DF.head(10)
    # top_predictions_for_user(DF)
    # grid_search(**args("NARRE_train_BPR"))

    #from BPR import test_bpr

    #recall, avg_recall, avg_precision, avg_hit_rate, avg_auc = test_bpr(
    #    **args("DeepCoNN_test_BPR"))

    # Stampiamo i risultati
    #print("Valore di Recall [ " + str(-recall) + " ]")
    #print("Valore di Media Recall [ " + str(-avg_recall) + " ]")
    #print("Valore di Precisione Media [ " + str(avg_precision) + " ]")
    # print("Valore di Hit Rate Media [ " + str(avg_hit_rate) + " ]")
    # print("Valore di AUC Media [ " + str(avg_auc) + " ]")
    # print("Valore di Predict Loss [ " + str(predict_loss) + " ]")
    # print("Valore di Test MSE [ " + str(test_mse) + " ]")
    # print("Valore di Test MAE [ " + str(test_mae) + " ]")
