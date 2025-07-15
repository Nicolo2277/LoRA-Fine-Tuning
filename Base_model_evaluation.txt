'''Base model evaluation script for story generation comparison'''
import os
import logging
import re
import numpy as np
import argparse
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import pandas as pd

# Logging setup:
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class BaseModelInference:
    '''Handles loading the base model and performing inference for comparison'''

    def __init__(self, base_model_name='distilgpt2'):
        '''
        base_model_name: name of the base model to evaluate
        '''
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        '''Load the base model and tokenizer'''
        logger.info(f'Loading the base model: {self.base_model_name}')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,  # accelerate inference
            device_map='auto'
        )
        self.model.eval()

        logger.info('Base model loaded successfully')
        return self.model, self.tokenizer

    def compute_loss(self, texts):
        '''Compute avg cross entropy loss'''

        batch_losses = []
        batch_perpl = []

        for text in texts:
            # Split the text to identify prompt and story parts
            if 'Story: ' in text:
                parts = text.split('Story: ', 1)
                if len(parts) == 2:
                    prompt = parts[0] + 'Story: '
                    story_part = parts[1]
                else:
                    prompt = 'Story: '
                    story_part = text
            else:
                prompt = ''
                story_part = text

            # Add EOS token to match training format
            full_text = text + self.tokenizer.eos_token

            # Tokenize the full text
            inputs = self.tokenizer(
                full_text,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get prompt length for masking
            if prompt:
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
                prompt_length = len(prompt_tokens)
            else:
                prompt_length = 0

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Create labels as in training
                input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
                labels = torch.full_like(input_ids, -100)  # Initialize all as -100

                # Only compute loss on story tokens
                if prompt_length < len(input_ids):
                    labels[prompt_length:] = input_ids[prompt_length:]

                # Compute cross entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[1:].contiguous()

                # Only compute loss where labels != -100
                valid_mask = (shift_labels != -100)

                if valid_mask.sum() > 0:
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    # Only average over valid tokens
                    valid_losses = losses[shift_labels.view(-1) != -100]
                    avg_loss = valid_losses.mean().item()
                    perpl = np.exp(avg_loss)
                else:
                    avg_loss = 0.0
                    perpl = 1.0

            batch_losses.append(avg_loss)
            batch_perpl.append(perpl)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return batch_losses, batch_perpl


class TestDataProcessor:
    '''Handles test dataset loading and preprocessing'''

    def __init__(self, tokenizer, max_length=400, min_length=20, max_tokens=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.max_tokens = max_tokens

    def clean_text(self, text):
        '''Clean and normalize the text'''
        text = text.strip()
        text = "".join(ch for ch in text if ch.isprintable() or ch == '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        return text

    def load_test_dataset(self):
        '''Load the test set as in the instructions'''
        full_dataset = load_dataset("skeskinen/TinyStories-GPT4", split="train")
        full_dataset = full_dataset.remove_columns(
                [c for c in full_dataset.column_names if c not in ['story', 'features']])
        assert len(full_dataset) == 2745100

        splits = full_dataset.train_test_split(test_size=10000, seed=42, shuffle=True)

        test_dataset_raw = splits["test"]
        assert len(test_dataset_raw) == 10000

        def preprocess_and_filter(examples):
            cleaned_stories = [self.clean_text(story) for story in examples['story']]
            tokenized = self.tokenizer(cleaned_stories, add_special_tokens=False)
            token_lengths = [len(tokens) for tokens in tokenized['input_ids']]

            return {
                'story': cleaned_stories,
                'token_length': token_lengths,
                'features': examples['features']
            }

        preprocessed_dataset = test_dataset_raw.map(
            preprocess_and_filter,
            batched=True,
            num_proc=1,
            batch_size=1000
        )

        # Exclude too short or too long stories (that the model cannot process)
        filtered_dataset = preprocessed_dataset.filter(
              lambda x: self.min_length <= x['token_length'] <= self.max_tokens
          )

        test_dataset = filtered_dataset

        logger.info(f'Test dataset size: {len(test_dataset)}')
        return test_dataset


def evaluate_model(inference_model, test_dataset, output_file=None, batch_size=8):
    '''Model evaluation using batched processing'''

    logger.info('Starting base model evaluation')
    results = []
    all_losses = []
    all_perpl = []

    for i in tqdm(range(0, len(test_dataset), batch_size), desc='Evaluating batches'):
        batch_end = min(i + batch_size, len(test_dataset))
        batch_data = test_dataset.select(range(i, batch_end))

        # Prepare batch data:
        batch_features = []
        batch_stories = []
        batch_full_texts = []

        for example in batch_data:
            raw_features = example['features'] if example['features'] else []
            features = [feature.strip().lower() for feature in raw_features]
            original_story = example['story']

            # Create the same format as used in training
            if isinstance(features, list) and len(features) > 0:
                features_str = ", ".join(features)
                prompt = f'Features: {features_str}\nStory: '
            else:
                prompt = 'Story: '

            full_text = prompt + original_story

            batch_features.append(features)
            batch_stories.append(original_story)
            batch_full_texts.append(full_text)

        # Compute losses for the batch
        batch_losses, batch_perpl = inference_model.compute_loss(batch_full_texts)
        all_losses.extend(batch_losses)
        all_perpl.extend(batch_perpl)

        # Store results
        for j, (features, story, loss, perpl) in enumerate(zip(batch_features, batch_stories, batch_losses, batch_perpl)):
            results.append({
                'features': features,
                'story': story,
                'loss': loss,
                'perplexity': perpl
            })

    # Calculate average loss
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    avg_perpl = np.mean(all_perpl) if all_perpl else 1.0
    logger.info(f'Base model average loss: {avg_loss:.4f}')
    logger.info(f'Base model average perplexity: {avg_perpl:.4f}')

    return results


def main():
    # Test parameter definition:
    seed = 42
    base_model = 'distilgpt2'
    batch_size = 8

    # Set random seed:
    torch.manual_seed(seed)

    # Initialize base model inference:
    inference_model = BaseModelInference(base_model)
    model, tokenizer = inference_model.load_model()


    logger.info('Running base model evaluation')

    # Load and process test dataset
    processor = TestDataProcessor(tokenizer)
    test_dataset = processor.load_test_dataset()

    # Run evaluation
    results = evaluate_model(
        inference_model,
        test_dataset,
        batch_size=batch_size
    )

    logger.info('Base model evaluation completed.')


if __name__ == "__main__":
    main()