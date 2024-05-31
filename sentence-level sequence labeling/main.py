import argparse
import os
import torch
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed, get_linear_schedule_with_warmup
from utils import data2batch, batch2idx, get_optimizer, cut_off, evaluate_batch_insts, read_data, build_label_idx, \
    write_results
from model import NNCRF

import faulthandler

faulthandler.enable()


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--max_no_incre', type=int, default=10,
                        help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    # model hyper parameter
    parser.add_argument('--transformer', type=str, default='bert-base-chinese')
    parser.add_argument('--tokenizer', type=str, default='bert-base-chinese')
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")

    # Path
    parser.add_argument('--data_dir', type=str, default="data/ECOM2022/")
    parser.add_argument('--train_file', type=str, default="train")
    parser.add_argument('--dev_file', type=str, default="dev")
    parser.add_argument('--test_file', type=str, default="test")
    parser.add_argument('--model_dir', type=str, default="model_files/")
    parser.add_argument('--model_folder', type=str, default="sentence-level sequence labeling", help="The name to save the model files")
    parser.add_argument('--result_dir', type=str, default="result/")

    # Data
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_token', type=int, default=4096)

    # Others
    parser.add_argument('--device', type=str, default="cuda:0",
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7', 'cuda:8', 'cuda:9'],
                        help="GPU/CPU devices")
    parser.add_argument('--retrain', type=bool, default=False)

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config, train_insts, dev_insts, test_insts=None):
    # Init model using config.
    model = NNCRF(config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    # Count the number of parameters.
    num_param = 0
    for idx in list(model.parameters()):
        try:
            num_param += idx.size()[0] * idx.size()[1]
        except IndexError:
            num_param += idx.size()[0]
    print(num_param)

    # Cut off
    train_insts = cut_off(train_insts, config.max_token, tokenizer, False)
    dev_insts = cut_off(dev_insts, config.max_token, tokenizer, False)
    test_insts = cut_off(test_insts, config.max_token, tokenizer, False)

    # Get instances.
    print("Number of train instances: %d" % len(train_insts))
    batched_data = data2batch(config.batch_size, train_insts)
    dev_batches = data2batch(config.batch_size, dev_insts, shffule=False)
    test_batches = data2batch(config.batch_size, test_insts, shffule=False)

    # Path to save op_extract_model.
    model_folder = config.model_folder
    model_dir = config.model_dir + model_folder + '_' + str(config.lr) + '_' + str(config.epoch) + '_' + str(config.warmup_ratio)
    model_path = model_dir + f"/lstm_crf.m"

    # If model exists, evaluate and save results.
    if os.path.exists(model_dir) and not config.retrain:
        print(f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
              f"to avoid override.")
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        paras = {}
        for name, para in model.named_parameters():
            paras[name] = para
        transition_matrix = pd.DataFrame(paras['inferencer.transition'].cpu().detach().numpy(), columns=['pad', 's', 'o', 'b', 'i', 'e', 'start', 'stop'])
        transition_matrix.to_csv(config.result_dir + 'transition.csv', index=False)
        _, predictions = evaluate_model(config, model, dev_batches, "test", dev_insts, tokenizer)
        write_results(config.result_dir + 'dev_pred', dev_insts, predictions)
        _, predictions = evaluate_model(config, model, test_batches, "test", test_insts, tokenizer)
        write_results(config.result_dir + 'test_pred', test_insts, predictions)
        return

    if os.path.exists(model_dir) and config.retrain:
        print(f"The folder model_files/{model_folder} exists. But we'll retrain from scratch.")

    # If model not exists. Create new dirs.
    print("[Info] The model will be saved to: %s.tar.gz" % model_folder)
    os.makedirs(model_dir, exist_ok=True)  # create model files. not raise error if exist.

    # Get optimizer.
    total_steps = (len(train_insts) // config.batch_size) * config.epoch \
        if len(train_insts) % config.batch_size == 0 else (len(train_insts) // config.batch_size + 1) * config.epoch
    optimizer = get_optimizer(config, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(config.warmup_ratio * total_steps), num_training_steps=total_steps)

    # Train model.
    best_dev = [-1, 0, -1]
    no_incre_dev = 0
    step_losses = []
    for i in tqdm(range(1, config.epoch + 1), desc="Epoch"):
        model.train()
        for index in tqdm(range(len(batched_data))):
            processed_batched_data = batch2idx(config, batched_data[index], tokenizer)
            loss = model(*processed_batched_data)
            step_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        with open(config.model_dir + 'loss_' + str(config.lr) + '.txt', 'w', encoding='utf-8') as f:
            for step_loss in step_losses:
                f.write(str(step_loss) + '\n')

        model.eval()
        dev_metrics, predictions = evaluate_model(config, model, dev_batches, "dev", dev_insts, tokenizer)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            torch.save(model.state_dict(), model_path)
            write_results(config.result_dir + 'dev_pred', dev_insts, predictions)
        else:
            no_incre_dev += 1
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev" % no_incre_dev)
            break
    print("The best dev during training: %.2f" % (best_dev[0]))

    # Begin test.
    if test_insts is not None:
        print("Final testing.")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        _, predictions = evaluate_model(config, model, test_batches, "test", test_insts, tokenizer)
        write_results(config.result_dir + 'test_pred', test_insts, predictions)


def evaluate_model(config, model: NNCRF, batch_insts_ids, name: str, insts, tokenizer):
    print(colored('[Begin evaluating ' + name + '.]', 'green'))
    metrics, metrics_e2e = np.asarray([0, 0, 0], dtype=int), np.zeros((1, 3), dtype=int)
    batch_idx = 0
    batch_size = config.batch_size
    predictions = []

    # Calculate metrics by batch.
    with torch.no_grad():
        for batch in tqdm(batch_insts_ids):
            # get instances list.
            one_batch_insts = insts[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            # get ner result.
            processed_batched_data = batch2idx(config, batch, tokenizer)
            batch_max_scores, batch_max_ids = model.decode(processed_batched_data)

            # evaluate ner result.
            # get the num of correctly predicted arguments, predicted arguments and gold arguments.
            metric, prediction = evaluate_batch_insts(one_batch_insts, batch_max_ids, processed_batched_data[-1],
                                                      processed_batched_data[-2], config.idx2labels)
            metrics += metric
            predictions += prediction

            batch_idx += 1
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    print("Opinion Extraction: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore),
          flush=True)
    print("Correctly predicted opinion: %d, Total predicted opinion: %d, Total golden opinion: % d" % (
    p, total_predict, total_entity))
    return [precision, recall, fscore], predictions


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    # Set seed.
    set_seed(args.seed)

    # Read train/test/dev data into instance.
    args.train_file = args.data_dir + args.train_file
    args.dev_file = args.data_dir + args.dev_file
    args.test_file = args.data_dir + args.test_file
    devs = read_data(args.dev_file, args.dev_num, map_iobes=True)
    tests = read_data(args.test_file, args.test_num, map_iobes=True)
    trains = read_data(args.train_file, args.train_num, map_iobes=True)

    # Get labels.
    args.label2idx = build_label_idx(trains + devs + tests)
    args.idx2labels = list(args.label2idx.keys())
    args.label_size = len(args.idx2labels)

    # Train model.
    train_model(args, trains, devs, tests)


if __name__ == "__main__":
    main()
