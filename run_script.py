# -*- coding:utf-8 -*-
import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl
from torch.nn import MSELoss

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertConfig
)

from models.modeling_span import Span_Detector
from models.modeling_type import Type_Classifier
from utils.data_utils import load_and_cache_examples, get_labels
# from utils.model_utils import mask_tokens, soft_frequency, opt_grad, get_hard_label, _update_mean_model_variables
from utils.eval import evaluate
from utils.config import config
# from utils.loss_utils import CycleConsistencyLoss

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "span": (Span_Detector, BertConfig, BertTokenizer),
    "type": (Type_Classifier, BertConfig, BertTokenizer),
    # "finetune": (TokenClassification, BertConfig, BertTokenizer)
}

torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, tokenizer, t_total, span_num_labels, type_num_labels_src, type_num_labels_tgt):
    model_class, config_class, _ = MODEL_CLASSES["span"]

    config = config_class.from_pretrained(
        args.span_model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model = model_class.from_pretrained(
        args.span_model_name_or_path,
        config=config,
        span_num_labels=span_num_labels,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model.to(args.device)
    
    model_class, config_class, _ = MODEL_CLASSES["type"]
    # config_class, model_class, _ = MODEL_CLASSES["student1"]
    # config_s1 = config_class.from_pretrained(
    #     args.student1_config_name if args.student1_config_name else args.student1_model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    config = config_class.from_pretrained(
        args.type_model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    type_model = model_class.from_pretrained(
        args.type_model_name_or_path,
        config=config,
        type_num_labels_src=type_num_labels_src,
        type_num_labels_tgt=type_num_labels_tgt, 
        device=args.device,
        domain=args.dataset,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    type_model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_span = [
        {
            "params": [p for n, p in span_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in span_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_span = AdamW(optimizer_grouped_parameters_span, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_span = get_linear_schedule_with_warmup(
        optimizer_span, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    optimizer_grouped_parameters_type = [
        {
            "params": [p for n, p in type_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in type_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_type = AdamW(optimizer_grouped_parameters_type, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_type = get_linear_schedule_with_warmup(
        optimizer_type, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     # [span_model, type_model], [optimizer_span, optimizer_type] = amp.initialize(
    #     #              [span_model, type_model], [optimizer_span, optimizer_type], opt_level=args.fp16_opt_level)
    #     span_model, optimizer_span = amp.initialize(
    #                  span_model, optimizer_span, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        span_model = torch.nn.DataParallel(span_model)
        type_model = torch.nn.DataParallel(type_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        span_model = torch.nn.parallel.DistributedDataParallel(
            span_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        type_model = torch.nn.parallel.DistributedDataParallel(
            type_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    span_model.zero_grad()
    type_model.zero_grad()

    return span_model, type_model, optimizer_span, scheduler_span, optimizer_type, scheduler_type
    # return span_model, optimizer_span, scheduler_span

def validation(args, span_model, type_model, tokenizer, id_to_label_span, pad_token_label_id, best_dev, best_test, best_dev_bio, best_test_bio,\
         global_step, t_total, epoch):
    best_dev, best_dev_bio, is_updated_dev = evaluate(args, span_model, type_model, tokenizer, \
        id_to_label_span, pad_token_label_id, best_dev, best_dev_bio, mode="dev", logger=logger, \
        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
    best_test, best_test_bio, is_updated_test = evaluate(args, span_model, type_model, tokenizer, \
        id_to_label_span, pad_token_label_id, best_test, best_test_bio, mode="test", logger=logger, \
        prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    # output_dirs = []
    # if args.local_rank in [-1, 0] and is_updated_dev:
    #     # updated_self_training_teacher = True
    #     path = os.path.join(args.output_dir, "checkpoint-best")
    #     logger.info("Saving model checkpoint to %s", path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     model_to_save = (
    #             span_model.module if hasattr(span_model, "module") else span_model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(path)
    #     tokenizer.save_pretrained(path)
    # # output_dirs = []
    # if args.local_rank in [-1, 0] and is_updated2:
    #     # updated_self_training_teacher = True
    #     path = os.path.join(args.output_dir+tors, "checkpoint-best-2")
    #     logger.info("Saving model checkpoint to %s", path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     model_to_save = (
    #             model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(path)
    #     tokenizer.save_pretrained(path)

    return best_dev, best_test, best_dev_bio, best_test_bio, is_updated_dev


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def share_loss(span_outputs, type_outputs, loss_funct, layers=3):
    # ((batch_size, seq, dim), ...) # Layer-0, 1, ...
    loss = 0.0
    for i in range(layers):
        loss += loss_funct(span_outputs[i], type_outputs[i])

    return loss

def train(args, train_dataset, train_dataset_meta, train_dataset_inter, id_to_label_span, id_to_label_type_src, id_to_label_type_tgt, tokenizer, pad_token_label_id):
    """ Train the model """
    # num_labels = len(labels)
    span_num_labels = len(id_to_label_span)
    type_num_labels_src = len(id_to_label_type_src)-1
    type_num_labels_tgt = len(id_to_label_type_tgt)-1
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    args.train_batch_size_meta = args.per_gpu_train_batch_size_meta * max(1, args.n_gpu)
    train_sampler_meta = RandomSampler(train_dataset_meta) if args.local_rank==-1 else DistributedSampler(train_dataset_meta)
    train_dataloader_meta = DataLoader(train_dataset_meta, sampler=train_sampler_meta, batch_size=args.train_batch_size_meta)
    # args.train_batch_size_inter = args.per_gpu_train_batch_size_inter * max(1, args.n_gpu)
    # train_sampler_inter = RandomSampler(train_dataset_inter) if args.local_rank==-1 else DistributedSampler(train_dataset_inter)
    # train_dataloader_inter = DataLoader(train_dataset_inter, sampler=train_sampler_inter, batch_size=args.train_batch_size_inter)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    span_model, type_model, optimizer_span, scheduler_span, \
    optimizer_type, scheduler_type = initialize(args, tokenizer, t_total, span_num_labels, type_num_labels_src, type_num_labels_tgt)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]
    best_dev_bio, best_test_bio = [0, 0, 0], [0, 0, 0]
    # meta_best_dev, meta_best_test = [0, 0, 0], [0, 0, 0]
    # s1_best_dev, s1_best_test = [0, 0, 0], [0, 0, 0]
    # s2_best_dev, s2_best_test = [0, 0, 0], [0, 0, 0]
    # t1_best_dev, t1_best_test = [0, 0, 0], [0, 0, 0]
    # t2_best_dev, t2_best_test = [0, 0, 0], [0, 0, 0]

    # self_learning_teacher_model1 = model_s1
    # self_learning_teacher_model2 = model_s2

    # softmax = torch.nn.Softmax(dim=1)
    # t_model1 = copy.deepcopy(model_s1)
    # t_model2 = copy.deepcopy(model_s2)

    # loss_regular = NegEntropy()
    loss_funct = MSELoss()

    # begin_global_step = len(train_dataloader)*args.begin_epoch//args.gradient_accumulation_steps
    iterator_meta = iter(cycle(train_dataloader_meta))
    # iterator_inter = iter(cycle(train_dataloader_inter))
    len_dataloader = len(train_dataloader)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            span_model.train()
            type_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            batch_meta = next(iterator_meta)
            batch_meta = tuple(t.to(args.device) for t in batch_meta)
            # batch_inter = next(iterator_inter)
            # batch_inter = tuple(t.to(args.device) for t in batch_inter)
            # p = float(step + epoch * len_dataloader) / t_total
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_bio": batch[2], "tgt": False, "reduction": "none"}
            outputs_span = span_model(**inputs)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_type": batch[3], "logits_bio": outputs_span[2], "tgt": False}
            outputs_type = type_model(**inputs)
            # loss1 = outputs_span[0]
            loss1 = span_model.loss(outputs_span[0], outputs_type[1], delta=args.delta_span)
            # loss2 = outputs_type[0]
            loss2 = type_model.loss(outputs_type[0], outputs_span[1], delta=args.delta_type)
            loss6 = share_loss(outputs_span[4], outputs_type[4], loss_funct)

            inputs = {"input_ids": batch_meta[0], "attention_mask": batch_meta[1], "labels_bio": batch_meta[2], "tgt": True, "reduction": "none"}
            outputs_span_meta = span_model(**inputs)

            inputs = {"input_ids": batch_meta[0], "attention_mask": batch_meta[1], "labels_type": batch_meta[3], "logits_bio": outputs_span_meta[2], "tgt": True}
            outputs_type_meta = type_model(**inputs)
            # loss3 = outputs_span_meta[0]
            loss3 = span_model.loss(outputs_span_meta[0], outputs_type_meta[1], delta=args.delta_span)
            permute_embed = span_model.adv_attack(outputs_span_meta[4][0], loss3, epsilon=args.mu)
            inputs = {"inputs_embeds": permute_embed, "attention_mask": batch_meta[1], "labels_bio": batch_meta[2], "tgt": True, "reduction": "mean"}
            outputs_span_meta_ = span_model(**inputs)
            loss31 = outputs_span_meta_[0] 
            # loss4 = outputs_type_meta[0]
            loss4 = type_model.loss(outputs_type_meta[0], outputs_span_meta[1], delta=args.delta_type)
            permute_embed = type_model.adv_attack(outputs_type_meta[4][0], loss4, epsilon=args.mu)
            inputs = {"inputs_embeds": permute_embed, "attention_mask": batch_meta[1], "labels_type": batch_meta[3], "tgt": True, "reduction": "mean"}
            outputs_type_meta_ = type_model(**inputs)
            loss41 = outputs_type_meta_[0] 

            loss7 = share_loss(outputs_span_meta[4], outputs_type_meta[4], loss_funct)

            # inputs = {"input_ids": batch_inter[0], "attention_mask": batch_inter[1], "labels_bio": batch_inter[2], "tgt": False, "reduction": "mean"}
            # outputs_span_inter = span_model(**inputs)
            # loss8 = outputs_span_inter[0]

            loss5 = type_model.mix_up(outputs_type[3], outputs_type_meta[3], batch[3], batch_meta[3], args.alpha, args.beta)

            # loss = outputs_span[0]+outputs_type[0]+outputs_span_meta[0]+outputs_type_meta[0]
            loss = loss1+loss2+loss3+loss4+0.1*loss5+0.1*(loss6+loss7)+0.1*(loss31+loss41)
            # loss = loss1+loss3+loss8

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer_span) as scaled_loss:
            #         scaled_loss.backward()
            #     # with amp.scale_loss(loss, optimizer_type) as scaled_loss:
            #     #     scaled_loss.backward()
            # else:
            loss.backward()

            tr_loss += loss.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_span), args.max_grad_norm)
                #     # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_type), args.max_grad_norm)
                # else:
                #     torch.nn.utils.clip_grad_norm_(span_model.parameters(), args.max_grad_norm)
                #     # torch.nn.utils.clip_grad_norm_(type_model.parameters(), args.max_grad_norm)
                optimizer_span.step()
                optimizer_type.step()
                scheduler_span.step()  # Update learning rate schedule
                scheduler_type.step()
                span_model.zero_grad()
                type_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        logger.info("***** training loss : %.4f *****", loss.item())
                        best_dev, best_test, best_dev_bio, best_test_bio, _ = validation(args, span_model, type_model, tokenizer, \
                            id_to_label_span, pad_token_label_id, best_dev, best_test, best_dev_bio, best_test_bio,\
                            global_step, t_total, epoch)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev, best_test)

    return results

def main():
    args = config()
    args.do_train = args.do_train.lower()
    args.do_test = args.do_test.lower()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    id_to_label_span, id_to_label_type_tgt, id_to_label_type_src = get_labels(args.data_dir, args.dataset)
    # num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = MODEL_CLASSES["span"][2].from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Loss = CycleConsistencyLoss(non_entity_id, args.device)

    # Training
    if args.do_train=="true":
        train_dataset, train_dataset_meta, train_dataset_inter = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        best_results = train(args, train_dataset, train_dataset_meta, train_dataset_inter,\
            id_to_label_span, id_to_label_type_src, id_to_label_type_tgt, tokenizer, pad_token_label_id)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # # Testing
    # if args.do_test=="true" and args.local_rank in [-1, 0]:
    #     best_test = [0, 0, 0]
    #     for tors in MODEL_NAMES:
    #         best_test = predict(args, tors, labels, pad_token_label_id, best_test)

# def predict(args, tors, labels, pad_token_label_id, best_test):
#     path = os.path.join(args.output_dir+tors, "checkpoint-best-2")
#     tokenizer = RobertaTokenizer.from_pretrained(path, do_lower_case=args.do_lower_case)
#     model = RobertaForTokenClassification_Modified.from_pretrained(path)
#     model.to(args.device)

#     # if not best_test:
   
#     # result, predictions, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_test, mode="test")
#     result, _, best_test, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_test, mode="test", \
#                                                         logger=logger, verbose=False)
#     # Save results
#     output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
#     with open(output_test_results_file, "w") as writer:
#         for key in sorted(result.keys()):
#             writer.write("{} = {}\n".format(key, str(result[key])))

#     return best_test
#     # Save predictions
#     # output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
#     # with open(output_test_predictions_file, "w") as writer:
#     #     with open(os.path.join(args.data_dir, args.dataset+"_test.json"), "r") as f:
#     #         example_id = 0
#     #         data = json.load(f)
#     #         for item in data: # original tag_ro_id must be {XXX:0, xxx:1, ...}
#     #             tags = item["tags"]
#     #             golden_labels = [labels[tag] for tag in tags]
#     #             output_line = str(item["str_words"]) + "\n" + str(golden_labels)+"\n"+str(predictions[example_id]) + "\n"
#     #             writer.write(output_line)
#     #             example_id += 1

if __name__ == "__main__":
    main()
