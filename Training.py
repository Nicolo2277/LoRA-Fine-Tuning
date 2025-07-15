###########################################################################################################
#Training script, runned on WSL on a GeForce MX 150 GPU on our local machine
'''
LoRA fine tuning
'''

import os
import logging
import re
import numpy as np
import argparse
from typing import Dict, List, Optional
import json
from pathlib import Path
import hashlib
import math

import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

#Logging setup:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WandbLoggingCallback(TrainerCallback):
    """Callback to handle wandb logging"""
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        self.train_losses = []
        self.train_perplexities = []
        self.eval_losses = []
        self.eval_perplexities = []

    def compute_perplexity(self, loss):
        perplexity = math.exp(loss)
        return perplexity

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.use_wandb:
            #Create a dictionary to hold additional metrics to log
            additional_logs = {}

            if 'train_loss' in logs:
                self.train_losses.append(logs['train_loss'])
                train_perplexity = self.compute_perplexity(logs['train_loss'])
                self.train_perplexities.append(train_perplexity)

                additional_logs['train/perplexity'] = train_perplexity

            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                eval_perplexity = self.compute_perplexity(logs['eval_loss'])
                self.eval_perplexities.append(eval_perplexity)

                additional_logs['eval/perplexity'] = eval_perplexity

            # Log additional metrics if any
            if additional_logs:
                wandb.log(additional_logs, step=state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        '''Called at the end of each epoch'''
        if self.use_wandb:
            wandb_logs = {'epoch_summary/epoch': state.epoch}
            if self.train_losses:
                epoch_train_loss = np.mean(self.train_losses)
                epoch_train_perplexity = np.mean(self.train_perplexities)
                wandb_logs.update({
                    'epoch_summary/train_loss': epoch_train_loss,
                    'epoch_summary/train_perplexity': epoch_train_perplexity,
                })

                if self.eval_losses:
                    epoch_eval_loss = np.mean(self.eval_losses)
                    epoch_eval_perplexity = np.mean(self.eval_perplexities)
                    wandb_logs.update({
                        'epoch_summary/eval_loss': epoch_eval_loss,
                        'epoch_summary/eval_perplexity': epoch_eval_perplexity,
                    })

                wandb.log(wandb_logs)

        #Clear losses for next epoch
        self.train_losses = []
        self.eval_losses = []
        self.train_perplexities = []
        self.eval_perplexities = []
        print('end of an epoch')

class StoryDataset:
    '''
    Class for dataset loading, cleaning and preprocessing
    '''

    def __init__(self, tokenizer, max_length=400, min_length=20, max_tokens=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.max_tokens = max_tokens

    def clean_text(self, text):
        '''
        Clean and normalize the text
        '''
        text = text.strip()
        #remove the non printable charachters except the new lines:
        text = "".join(ch for ch in text if ch.isprintable() or ch == '\n')
        #normalize spaces, but keeping single newlines:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        return text

    def load_and_prepare_dataset(self, path, val_size, seed=42):

        logger.info(f'Loading the dataset from {path}')
        full_dataset = load_from_disk(path)

        #remove unnecessary cols:
        full_dataset = full_dataset.remove_columns(
            [c for c in full_dataset.column_names if c not in ['story', 'features']]
        )

        #clean text and calculate token lengths:
        def preprocessing_and_filter(examples):
            #clean text:
            cleaned_stories = [self.clean_text(story) for story in examples['story']]
            #calculate token lengths for filtering purposes:
            tokenized = self.tokenizer(cleaned_stories, add_special_tokens=False)
            token_lengths = [len(tokens) for tokens in tokenized['input_ids']]

            return{
                'story':cleaned_stories,
                'token_length':token_lengths,
                'features':examples['features']
            }

        processed_dataset = full_dataset.map(preprocessing_and_filter,
                                             batched=True,
                                             num_proc=1,
                                             batch_size=1000)
        #filter by token lengths:
        filtered_dataset = processed_dataset.filter(
            lambda x: self.min_length <= x['token_length'] <= self.max_tokens
        )

        logger.info(f'Original dataset size: {len(full_dataset)}')
        logger.info(f'After filtering: {len(filtered_dataset)}')

        #create splits
        split = filtered_dataset.train_test_split(test_size=val_size, seed=seed) #seed to ensure reproducibility
        datasets = {
        'train': split['train'],
        'validation': split['test']
        }

        logger.info(f"Train dataset size: {len(split['train'])}")
        logger.info(f"Validation dataset size: {len(split['test'])}")

        return datasets

    def preprocess_for_training(self, examples):

        #Process each example individually
        processed_examples = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for i in range(len(examples['story'])):
            # Create prompt for this example
            features = examples['features'][i]
            if isinstance(features, list) and len(features) > 0:
                features_str = ", ".join(features)
                prompt = f'Features: {features_str}\nStory: '
            else:
                prompt = 'Story: '

            # Combine prompt and story
            full_text = prompt + examples['story'][i] + self.tokenizer.eos_token

            #Tokenize it
            tokenized = self.tokenizer(
                full_text,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )

            # Get prompt tokens to know where to start learning
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens)

            # Create labels
            input_ids = tokenized['input_ids']
            labels = [-100] * len(input_ids)  # Initialize all as -100

            # Only learn to predict tokens after the prompt
            for j in range(prompt_length, len(input_ids)):
                labels[j] = input_ids[j]

            processed_examples['input_ids'].append(input_ids)
            processed_examples['attention_mask'].append(tokenized['attention_mask'])
            processed_examples['labels'].append(labels)

        return processed_examples


    def load_and_preprocess_datasets(self, path, val_size, seed=42):
        # Load raw datasets
        datasets = self.load_and_prepare_dataset(path, val_size, seed)
        # Preprocess for training
        preprocessed_datasets = {}
        for split_name in datasets:
            logger.info(f'Preprocessing {split_name} dataset')
            preprocessed_datasets[split_name] = datasets[split_name].map(
                self.preprocess_for_training,
                batched=True,
                batch_size=100, #smaller so it doesnt go OOM
                num_proc=1,
                remove_columns=['story', 'features', 'token_length']
            )
        return preprocessed_datasets

class ModelTrainer:
    """Handles LoRA setup and training"""

    def __init__(self, model_name='distilgpt2'):
        self.model_name = model_name
        self.config = self._get_default_config()
        self.tokenizer = None
        self.model = None

    def _get_default_config(self): #parameters for LoRA fine tuning
        return {
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ['c_attn', 'c_proj', 'c_fc'],
            'learning_rate': 2e-4,
            'batch_size': 4,  #small because of limited GPU power
            'gradient_accumulation_steps': 4,  #we increased accumulation to balance the small batch size
            'num_epochs': 5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0
        }

    def setup_tokenizer(self):
        """Initialize and configure tokenizer for distilledgpt2 model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # left padding
        self.tokenizer.padding_side = 'left'
        return self.tokenizer

    def setup_model(self):
        """Initialize model with LoRA"""
        #load the base pretrained model:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True  #compatibility purposes
        )

        #LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, #indicates that we are adapting a causal language model
            inference_mode=False,
            r=self.config['lora_r'], #r is the rank of low-rank update
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=['c_attn', 'c_proj'], #we only insert LoRA adapters into c_attn and c_proj layers
            bias='none'
        )

        #apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters() #to see how many parameters we are training

        return self.model

    def create_trainer(self, datasets, output_dir, use_wandb=False):

        def data_collator(features):
            """Custom data collator to assure padding consistency"""

            # Find the maximum length in this batch
            max_length = max(len(f['input_ids']) for f in features)

            batch = {
                'input_ids': [],
                'attention_mask': [],
                'labels': []
            }

            for feature in features:
                input_ids = feature['input_ids']
                attention_mask = feature['attention_mask']
                labels = feature['labels']

                # Pad to max_length if needed
                current_length = len(input_ids)
                if current_length < max_length:
                    pad_length = max_length - current_length
                    # Use tokenizer's pad_token_id for padding
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

                    # Right padding
                    input_ids = input_ids + [pad_token_id] * pad_length
                    attention_mask = attention_mask + [0] * pad_length
                    labels = labels + [-100] * pad_length

                batch['input_ids'].append(input_ids)
                batch['attention_mask'].append(attention_mask)
                batch['labels'].append(labels)

            # Convert to tensors
            batch['input_ids'] = torch.tensor(batch['input_ids'], dtype=torch.long)
            batch['attention_mask'] = torch.tensor(batch['attention_mask'], dtype=torch.long)
            batch['labels'] = torch.tensor(batch['labels'], dtype=torch.long)

            return batch

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            num_train_epochs=self.config['num_epochs'],
            learning_rate=self.config['learning_rate'],
            warmup_ratio=self.config['warmup_ratio'],
            weight_decay=self.config['weight_decay'],
            max_grad_norm=self.config['max_grad_norm'],

            #logging and evaluation
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',

            #Optimization settings
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,  #don't drop last batch
            remove_unused_columns=False,  #keep all columns

            #checkpoints
            save_total_limit=1, #save 1 checkpoints
            save_only_model=True, #save only the model's weigths

            #wandb:
            report_to='wandb' if use_wandb else 'none',
        )

        callbacks=[]
        if use_wandb:
            callbacks.append(WandbLoggingCallback(use_wandb=use_wandb))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=data_collator,
            callbacks=callbacks
        )

        return trainer


def main():
    #Training arguments
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for story generation task')
    parser.add_argument('--model_name', type=str, default='distilgpt2', help='Base model name')
    parser.add_argument('--dataset', type=str, default='./final_datasets/sampled_train', help='Cache dataset path')
    parser.add_argument('--output_dir', type=str, default='./lora-results', help='Output directory')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use Weights & Biases')
    parser.add_argument('--val_size', type=int, default=1000, help='Size of the validation set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_length', type=int, default=400, help='Maximum sequence length')

    args = parser.parse_args()

    #initialize trainer
    trainer_obj = ModelTrainer(args.model_name)
    tokenizer = trainer_obj.setup_tokenizer()
    model = trainer_obj.setup_model()

    processor = StoryDataset(tokenizer, max_length=args.max_length)

    #dataset loading and preprocessing
    datasets = processor.load_and_preprocess_datasets(
        path=args.dataset,
        val_size=args.val_size,
        seed=args.seed
    )
    logger.info('Datasets loaded successfully')

    #initialize wandb:
    if args.use_wandb:
        run_name = f"lora-r{trainer_obj.config['lora_r']}"
        wandb.init(project='lora-story-generation',
                   name=run_name)

    #create trainer:
    trainer = trainer_obj.create_trainer(datasets, args.output_dir, args.use_wandb)

    logger.info('Starting training ')
    logger.info(f"Train dataset size: {len(datasets['train'])}")
    logger.info(f"Validation dataset size: {len(datasets['validation'])}")

    #start training:
    train_result = trainer.train()
    logger.info(f'Training completed, final loss: {train_result.training_loss}')

    #save model
    logger.info('Saving model')
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()