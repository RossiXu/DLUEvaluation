import json
import random

import torch
from termcolor import colored
import numpy as np
import pandas as pd
from torch import optim

PAD, START, STOP = "<PAD>", "<START>", "<STOP>"


def read_data(file, number=5, map_iobes=True):
    print("Reading " + file + " file.")
    insts = []

    # Read document.
    with open(file + '.doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # Read annotation.
    try:
        with open(file + '.ann.json', 'r', encoding='utf-8') as f:
            opinions = json.load(f)
    except FileNotFoundError:
        opinions = []
        print(colored('[There is no ' + file + '.ann.json.]', 'red'))

    for doc in docs:
        doc_id = doc['id']
        annotation = [annotation for annotation in opinions if annotation['id'] == doc_id][0]
        sents = doc['document']
        labels = annotation['labels']
        inst = [sents, labels]
        insts.append(inst)

    if number > 0:
        insts = insts[:number]

    if map_iobes:
        for idx, inst in enumerate(insts):
            labels = inst[1]
            new_labels = []
            for label_idx, label in enumerate(labels):
                new_label = label
                if label == 'B' and (label_idx + 1 == len(labels) or labels[label_idx + 1] != 'I'):
                    new_label = 'S'
                if label == 'I' and (label_idx + 1 == len(labels) or labels[label_idx + 1] != 'I'):
                    new_label = 'E'
                new_labels.append(new_label)
            insts[idx][1] = new_labels

    print("Number of documents: {}".format(len(insts)))
    return insts


def cut_off(insts, max_token, tokenizer, save=True):
    if max_token > 0:
        new_insts = []
        for idx, inst in enumerate(insts):
            new_sents = []
            new_labels = []
            doc_token_num = 0
            last_sent_idx = 0
            assert len(inst[0]) == len(inst[1])
            for sent_idx, sent in enumerate(inst[0]):
                sent_token_num = len(tokenizer.tokenize(sent)) + 2

                if sent_token_num + doc_token_num > max_token:
                    new_insts.append([new_sents, new_labels])
                    new_sents = [sent]
                    new_labels = [inst[1][sent_idx]]
                    doc_token_num = sent_token_num
                    last_sent_idx = sent_idx - 1
                    if not save:
                        last_sent_idx = len(inst[0]) - 1
                        break
                else:
                    new_sents.append(sent)
                    new_labels.append(inst[1][sent_idx])
                    doc_token_num += sent_token_num

            if last_sent_idx < len(inst[0]) - 1:
                new_insts.append([inst[0][last_sent_idx:], inst[1][last_sent_idx:]])
    else:
        new_insts = insts
    return new_insts


def build_label_idx(insts):
    label2idx = {}
    label2idx[PAD] = len(label2idx)
    for inst in insts:
        for label in inst[1]:
            if label not in label2idx.keys():
                label2idx[label] = len(label2idx)
    label2idx[START] = len(label2idx)
    label2idx[STOP] = len(label2idx)
    print('label2idx: {}'.format(label2idx))
    return label2idx


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], 1, vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def data2batch(batch_size, insts, shffule=True):
    """
    List of instances -> List of batches
    """
    if shffule:
        insts.sort(key=lambda x: len(x[0]))
    train_num = len(insts)
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(one_batch_insts)
    if shffule:
        random.shuffle(batched_data)
    return batched_data


def batch2idx(config, insts, tokenizer):
    """
    batching these instances together and return tensors.
    :return
        doc_len: Shape: (batch_size), the length of each paragraph in a batch.
        sent_tensor: Shape: (batch_size, max_seq_len, max_token_num)
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_size = len(insts)
    word_pad_idx = tokenizer.pad_token_id

    # word -> id
    doc_ids = []  # [bs, doc_token_len]
    cls_locs = []
    for inst in insts:
        doc_subword = []
        cls_loc = []
        for sent in inst[0]:
            cls_loc.append(len(doc_subword))
            doc_subword += [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]
        doc_id = tokenizer.convert_tokens_to_ids(doc_subword)
        doc_ids.append(doc_id)
        cls_locs.append(cls_loc)

    # label -> id
    doc_sent_num = [len(inst[0]) for inst in insts]
    label_seq_tensor = torch.zeros((batch_size, max(doc_sent_num)), dtype=torch.long)
    for idx in range(batch_size):
        try:
            label_ids = [config.label2idx[label] for label in insts[idx][1]]
        except KeyError:
            print('Unseen Keys!')
            print(insts[idx])
        label_seq_tensor[idx, :doc_sent_num[idx]] = torch.LongTensor(label_ids)

    # padding
    # pad sent_num -> max_doc_sent_num
    for idx, doc_id in enumerate(doc_ids):
        pad_sent_num = max(doc_sent_num) - len(cls_locs[idx])
        for i in range(pad_sent_num):
            cls_locs[idx].append(len(doc_ids[idx]))
            doc_ids[idx].append(tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
    # pad token_num -> max_doc_token_num
    attention_masks = []
    doc_token_num = [len(doc_id) for doc_id in doc_ids]
    for idx, doc_id in enumerate(doc_ids):
        pad_token_num = max(doc_token_num) - len(doc_id)

        attention_mask = [1] * len(doc_id)
        attention_mask.extend([0] * pad_token_num)
        for sent_idx in cls_locs[idx]:
            attention_mask[sent_idx] = 2  # Longformer global attention
        attention_masks.append(attention_mask)

        doc_ids[idx].extend([word_pad_idx] * pad_token_num)

    # list to tensor
    doc_tensor = torch.LongTensor(doc_ids).to(config.device)
    label_tensor = label_seq_tensor.to(config.device)
    cls_tensor = torch.LongTensor(cls_locs).to(config.device)
    doc_len_tensor = torch.LongTensor(doc_sent_num).to(config.device)
    attention_mask_tensor = torch.LongTensor(attention_masks).to(config.device)

    return doc_tensor, attention_mask_tensor, cls_tensor, doc_len_tensor, label_tensor


def old_batch2idx(config, insts, tokenizer):
    """
    batching these instances together and return tensors.
    :return
        doc_len: Shape: (batch_size), the length of each paragraph in a batch.
        sent_tensor: Shape: (batch_size, max_seq_len, max_token_num)
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_size = len(insts)
    word_pad_idx = tokenizer.pad_token_id

    doc_sents = [inst[0] for inst in insts]  # [[inst1_sents], [inst2_sents], ...]
    max_doc_len = max([len(doc_sent) for doc_sent in doc_sents])
    doc_lens = torch.LongTensor(list(map(lambda inst: len(inst[0]), insts)))

    # word -> id
    doc_sent_ids = []
    max_token_len = 0
    for idx, doc in enumerate(doc_sents):
        sent_ids = [tokenizer.encode_plus(sent).input_ids for sent in doc]  # doc_len * token_num
        max_token_len = max(max_token_len, max([len(sent) for sent in sent_ids]))
        doc_sent_ids.append(sent_ids)

    # padding: batch_size, max_doc_len, max_token_len
    for doc_idx, doc_sent_id in enumerate(doc_sent_ids):
        # pad sent to max_token_len
        for sent_idx, sent_id in enumerate(doc_sent_id):
            pad_token_num = - len(sent_id) + max_token_len
            doc_sent_ids[doc_idx][sent_idx].extend([word_pad_idx]*pad_token_num)
        # pad doc to max_doc_len
        pad_sent_num = max_doc_len - len(doc_sent_id)
        for i in range(pad_sent_num):
            doc_sent_ids[doc_idx].append([word_pad_idx]*max_token_len)

    # label -> id
    label_seq_tensor = torch.zeros((batch_size, max_doc_len), dtype=torch.long)
    for idx in range(batch_size):
        label_ids = [config.label2idx[label] for label in insts[idx][1]]
        label_seq_tensor[idx, :doc_lens[idx]] = torch.LongTensor(label_ids)

    # list to tensor
    sent_tensor = torch.LongTensor(doc_sent_ids).to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    doc_len_tensor = doc_lens.to(config.device)

    return doc_len_tensor, sent_tensor, label_seq_tensor


def get_optimizer(config, model):
    """
    Method to get optimizer.
    """
    base_params = list(map(id, model.encoder.transformer.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": config.lr},
        {"params": model.encoder.transformer.parameters(), "lr": config.backbone_lr},
    ]

    if config.optimizer.lower() == "sgd":
        print(colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params)
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)


def use_ibo(label):
    if label.startswith('E'):
        label = 'I'
    elif label.startswith('S'):
        label = 'B'
    return label


def write_results(filename, insts, predictions):
    """
    Save results.
    Each json instance is an opinion.
    """
    doc_sents = []
    doc_gold_labels = []
    doc_pred_labels = []
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, inst in enumerate(insts):
            doc_sents += inst[0]
            doc_gold_labels += inst[1]
            doc_pred_labels += predictions[idx]
            doc_sents += ['']
            doc_gold_labels += ['']
            doc_pred_labels += ['']
    data = np.array(list(zip(doc_sents, doc_gold_labels, doc_pred_labels)))
    pd_data = pd.DataFrame(data, columns=['sent', 'gold_label', 'pred_label'])
    pd_data.to_csv(filename + '.csv', index=False)