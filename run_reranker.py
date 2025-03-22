# -*- coding: utf-8 -*-
# Fine-tuning script for CodeBERT Reranker

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer)

from model import CodeBERTReranker
from utils import (convert_examples_to_features, RerankerProcessor,
                  compute_reranker_metrics)

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and \
            os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * 
                (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_mrr = 0.0  # Track best MRR for model saving
    
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # For reproducibility
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            if args.reranker_type == "pointwise":
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels': batch[3]}
                outputs = model(**inputs)
                loss = outputs[0] if args.reranker_type == "pointwise" else outputs['loss']
            else:  # pairwise
                # Unpack batch for pairwise inputs (query with positive and negative examples)
                inputs = {
                    'pos_input_ids': batch[0],
                    'pos_attention_mask': batch[1],
                    'pos_token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'neg_input_ids': batch[3],
                    'neg_attention_mask': batch[4],
                    'neg_token_type_ids': batch[5] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels': batch[6] if len(batch) > 6 else None
                }
                outputs = model(**inputs)
                loss = outputs['loss']

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Save model
                    model_to_save = model.module if hasattr(model, 'module') else model
                    if hasattr(model_to_save, 'save_pretrained'):
                        model_to_save.save_pretrained(output_dir)
                    else:
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                    
                    # Save tokenizer and training args
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    
                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
                
        # Evaluate after each epoch
        if args.local_rank in [-1, 0] and args.evaluate_during_training:
            results = evaluate(args, model, tokenizer, prefix=f"epoch-{_}")
            for key, value in results.items():
                tb_writer.add_scalar(f'eval_{key}', value, _)
                
            # Save best model based on MRR
            if results.get('mrr', 0) > best_mrr:
                best_mrr = results['mrr']
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save model
                model_to_save = model.module if hasattr(model, 'module') else model
                if hasattr(model_to_save, 'save_pretrained'):
                    model_to_save.save_pretrained(output_dir)
                else:
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                
                # Save tokenizer and training args
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving best model checkpoint with MRR: %s to %s", best_mrr, output_dir)
                
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    """Evaluate the model on the dev/test set"""
    eval_output_dir = args.output_dir
    
    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Evaluate!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    all_scores = []
    all_labels = []
    all_query_ids = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            if args.reranker_type == "pointwise":
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None}
                
                # Get scores
                outputs = model.get_score(**inputs)
                scores = outputs.detach().cpu().numpy()
                
                # Get labels
                labels = batch[3].detach().cpu().numpy()
                
                # Get query IDs (if available)
                if len(batch) > 4:
                    query_ids = batch[4].detach().cpu().numpy()
                else:
                    query_ids = np.zeros_like(labels)
                
            else:  # For pairwise, during evaluation we still use pointwise scoring
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None}
                
                # Get scores
                scores = model.get_score(**inputs).detach().cpu().numpy()
                
                # Get labels
                labels = batch[3].detach().cpu().numpy()
                
                # Get query IDs (if available)
                if len(batch) > 4:
                    query_ids = batch[4].detach().cpu().numpy()
                else:
                    query_ids = np.zeros_like(labels)
            
            all_scores.append(scores)
            all_labels.append(labels)
            all_query_ids.append(query_ids)
    
    # Concatenate results
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_query_ids = np.concatenate(all_query_ids, axis=0)
    
    # Compute metrics
    metrics = compute_reranker_metrics(all_scores, all_labels, all_query_ids)
    
    for key, value in metrics.items():
        results[key] = value
    
    # Log results
    output_eval_file = os.path.join(eval_output_dir, "eval_results_{}.txt".format(prefix))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False):
    """Load and preprocess the reranking dataset"""
    # Create processor
    processor = RerankerProcessor()
    
    # Get examples
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir, args.dev_file)
    else:
        examples = processor.get_train_examples(args.data_dir, args.train_file)
    
    # Convert to features
    features = convert_examples_to_features(
        examples,
        args.max_seq_length,
        tokenizer,
        args.reranker_type,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.pad_token_id,
        cls_token_segment_id=0,
        pad_token_segment_id=0
    )
    
    # Convert to tensor dataset
    if args.reranker_type == "pointwise":
        # Input IDs, attention mask, token type IDs, labels, (optional) query IDs
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        
        if hasattr(features[0], 'query_id'):
            all_query_ids = torch.tensor([f.query_id for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_query_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
            
    else:  # pairwise
        # Positive examples
        all_pos_input_ids = torch.tensor([f.pos_input_ids for f in features], dtype=torch.long)
        all_pos_attention_mask = torch.tensor([f.pos_attention_mask for f in features], dtype=torch.long)
        all_pos_token_type_ids = torch.tensor([f.pos_token_type_ids for f in features], dtype=torch.long)
        
        # Negative examples
        all_neg_input_ids = torch.tensor([f.neg_input_ids for f in features], dtype=torch.long)
        all_neg_attention_mask = torch.tensor([f.neg_attention_mask for f in features], dtype=torch.long)
        all_neg_token_type_ids = torch.tensor([f.neg_token_type_ids for f in features], dtype=torch.long)
        
        # Create dummy labels (all 1s since pos should be ranked higher than neg)
        all_labels = torch.ones(len(features), dtype=torch.long)
        
        dataset = TensorDataset(
            all_pos_input_ids, all_pos_attention_mask, all_pos_token_type_ids,
            all_neg_input_ids, all_neg_attention_mask, all_neg_token_type_ids,
            all_labels
        )
    
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data directory")
    parser.add_argument("--model_type", default="roberta", type=str, required=True,
                        help="Model type: roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name (e.g., 'microsoft/codebert-base')")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--reranker_type", default="pointwise", type=str, required=True,
                        choices=["pointwise", "pairwise"],
                        help="Reranker type: pointwise or pairwise")

    # Data parameters
    parser.add_argument("--train_file", default="train.txt", type=str,
                        help="The training data file name")
    parser.add_argument("--dev_file", default="dev.txt", type=str,
                        help="The development data file name")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization")
    
    # Training parameters
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation on the dev set")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps")
    parser.add_argument("--margin", default=0.3, type=float,
                        help="Margin for pairwise ranking loss")
    
    # Logging and saving parameters
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps")
    
    # System parameters
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend
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
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training downloads the model & vocab
        torch.distributed.barrier()

    # Create config
    config = RobertaConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=2,  # Binary classification for pointwise reranking
        finetuning_task="reranking"
    )
    
    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    
    # Create model
    model = CodeBERTReranker(
        args.model_name_or_path, 
        config=config,
        margin=args.margin,
        reranker_type=args.reranker_type
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training downloads the model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save model
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            if hasattr(model_to_save, 'save_pretrained'):
                model_to_save.save_pretrained(args.output_dir)
            else:
                # For custom models that don't have save_pretrained
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
            
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # Load a trained model and evaluate
        logger.info("Evaluate the following checkpoint: %s", args.output_dir)
        
        model = CodeBERTReranker(
            args.output_dir,
            reranker_type=args.reranker_type,
            margin=args.margin
        )
        model.to(args.device)
        
        result = evaluate(args, model, tokenizer, prefix="final")
        results.update(result)

    return results

if __name__ == "__main__":
    main()