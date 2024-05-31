# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import numpy as np

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
import evaluate
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from linformer import LinformerConfig, LinformerForSequenceClassification

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="train model",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="eval model",
    )
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="continue train",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--initializer_range", type=float, default=0.02, help="parameters initializer range"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None and args.test_file is None:
        raise ValueError("Need either a task name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.info(args)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = (args.train_file if args.train_file is not None else args.test_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    # raw_datasets["train"]= load_dataset(extension, data_files=data_files,split="train[:400]")
    # print(raw_datasets)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = raw_datasets["train"].unique("label") if args.do_train else raw_datasets["test"].unique(
        "label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    # print(num_labels)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if "linformer" not in args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        # from transformers import BertForSequenceClassification
        # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, num_hidden_layers=3)
        # model = BertForSequenceClassification(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained("/shared_home/guanxinyan2022/model/roberta-base", use_fast=not args.use_slow_tokenizer)
        config = LinformerConfig(num_hidden_layers=6, hidden_size_k=256,
                                 parameter_sharing="layerwise", num_labels=num_labels,
                                 seq_length=args.max_length, vocab_size=len(tokenizer))
        model = LinformerForSequenceClassification(config)

    # Preprocessing the datasets

    from torchinfo import summary

    print(summary(model, input_size=(args.per_device_train_batch_size,args.max_length), device="cpu",dtypes=[torch.LongTensor],depth=7))
    # exit()


    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    if args.do_train:
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    else:
        non_label_column_names = [name for name in raw_datasets["test"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.

    label_to_id = {v: i for i, v in enumerate(label_list)}

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names if args.do_train else raw_datasets["test"].column_names,
            desc="Running tokenizer on dataset",
        )
    if args.do_train:
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    if args.do_eval:
        test_dataset = processed_datasets["test"]
        # Log a few random samples from the test set:
        for index in random.sample(range(len(test_dataset)), 3):
            logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    if args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.do_eval:
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    if args.do_train:
        train_dataloader, eval_dataloader = accelerator.prepare(
            train_dataloader, eval_dataloader
        )
    if args.do_eval:
        test_dataloader = accelerator.prepare(test_dataloader)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    if args.do_train:
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        cur_epoch=0
        cur_step=0
        if args.continue_train:
            logger.info(f'load from {args.model_name_or_path}')
            model_data=torch.load(args.model_name_or_path+'/optim.ckpt')
            lr_scheduler.load_state_dict(model_data['lr_scheduler'])
            optimizer.load_state_dict(model_data['optimizer'])
            cur_epoch=model_data['epoch']
            cur_step=model_data['step']
            logger.info(f'already train {cur_epoch+1} epoch')

    # Get the metric function

    metric = evaluate.load("accuracy")
    print(model)

    if args.do_train:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        best_dev_acc = 0

        # wandb.init(project='linformer-classify',
        #            config=args)
        # wandb.watch(model,log="all",log_graph=True)

        # import ipdb
        # ipdb.set_trace()
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                # if step==len(train_dataloader)-1:
                #     wandb.log({'loss':loss.item()},commit=False)
                # else:
                #     wandb.log({'loss':loss.item()})
                progress_bar.set_postfix(loss=loss.item(),lr=lr_scheduler.get_last_lr()[0])
                accelerator.backward(loss)

                # if step%500==0:
                #     with open(f'tmp/step{step}.txt', 'w') as f:
                #         for name, ele in model.named_parameters():
                #             f.write(name + '\n')
                #             f.write(str(ele.data) + '\n')
                #             f.write(str(ele.grad) + '\n')

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)
                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch["labels"]),
                    )
            eval_metric = metric.compute()
            logger.info(f"epoch {epoch} dev: {eval_metric}")
            if args.do_eval:
                model.eval()
                with torch.no_grad():
                    for step, batch in enumerate(test_dataloader):
                        outputs = model(**batch)
                        predictions = outputs.logits.argmax(dim=-1)
                        metric.add_batch(
                            predictions=accelerator.gather(predictions),
                            references=accelerator.gather(batch["labels"]),
                        )

                test_metric = metric.compute()
                logger.info(f"test accuracy: {test_metric}")
            # wandb.log({'dev_acc':eval_metric,'test_acc':test_metric})
            if best_dev_acc < eval_metric['accuracy']:
                logger.info(f"update ckp in epoch {epoch}")
                best_dev_acc = eval_metric['accuracy']
                args.bes_dev_acc = best_dev_acc
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                torch.save({'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'step': completed_steps+cur_step,
                            'epoch': epoch}, args.output_dir + '/optim.ckpt')
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                args_dic = vars(args)
                # print(args_dic)
                args_dic['lr_scheduler_type'] = str(args_dic['lr_scheduler_type'])
                with open(args.output_dir + '/args.json', 'w') as f:
                    json.dump(args_dic, f, indent=2)

                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                if args.do_eval:
                    with open(args.output_dir + '/result.json', 'w') as f:
                        test_metric["epoch"] = epoch
                        json.dump(test_metric, f, indent=2)

        # wandb.finish()



    elif args.do_eval:
        total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

        logger.info("***** Running test *****")
        logger.info(f"  Num examples = {len(test_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_dataloader)):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

        eval_metric = metric.compute()
        logger.info(f"test accuracy: {eval_metric}")
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        with open(args.output_dir + '/result.json', 'w') as f:
            json.dump(eval_metric, f, indent=2)


if __name__ == "__main__":
    main()
