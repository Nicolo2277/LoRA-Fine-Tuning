'''LoRA inference script for story generation'''
import os
import logging
import random
import re
import numpy as np
import argparse
import json
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig #Parameter Efficient FineTuning library (in this project we use LoRA)
from tqdm.notebook import tqdm
import pandas as pd
import gdown
COLAB_AVAILABLE = True

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
#Download nltk data


try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt_tab')

#Logging setup:
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class StoryQualityFilter:
    '''Filters stories based on quality metrics'''

    def __init__(self):
        #Common bad patterns to filter out
        self.bad_patterns = [
            r'^(.{1,10})\1{3,}',  #repeated short patterns
            r'[^\w\s]{10,}',  #too many special characters
            r'\b\w{50,}\b',  #extremely long words
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.bad_patterns]

    def calculate_quality_score(self, story):
        '''Calculate quality score (0-1) for a story'''
        if not story or len(story.strip()) < 10:
            return 0.0

        score = 0.5

        #Decrease the score if we encounter bad patterns
        for pattern in self.compiled_patterns:
            if pattern.search(story):
                score -= 0.2

        words = story.split()
        sentences = story.split('.')

        #Penalize very short or very long stories
        word_count = len(words)
        if word_count < 20:
            score -= 0.4
        elif word_count > 2000:
            score -= 0.2

        #Check sentence variety
        if len(sentences) > 1:
            avg_sentence_length = word_count / len(sentences)
            if 5 <= avg_sentence_length <= 30:
                score += 0.1

        #Check vocabulary diversity
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        vocab_ratio = unique_words / max(word_count, 1)
        if vocab_ratio > 0.3:
            score += 0.2
        elif vocab_ratio < 0.1:
            score -= 0.2

        #Plus if dialogue is present
        if '"' in story or "'" in story:
            score += 0.1

        return max(0.0, min(1.0, score))


class EvaluationMetrics:
  '''Compute some evaluation metrics (self.bleu_score, quality score, novelty)'''
  def __init__(self, training_samples=100):
    self.smoothing = SmoothingFunction
    self.quality_filter = StoryQualityFilter()
    self.training_samples = training_samples

  def preprocess_text(self, text):
    '''Clean and tokenize text'''
    text = text.strip()
    text = "".join(ch for ch in text if ch.isprintable() or ch == '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    tokens = word_tokenize(text.lower())
    return tokens

  def self_bleu(self, texts, max_n=4):
      '''Compute self_bleu metric, it measures how similar generated texts are to each other'''
      bleu_scores = []
      for i, text in enumerate(texts):
        #We use all other texts as reference for the current one:
        references = [self.preprocess_text(texts[j]) for j in range(len(texts)) if j != i]
        candidate = self.preprocess_text(text)

        bleu = sentence_bleu(
            references,
            candidate,
            weights=[1.0/max_n] * max_n,
            smoothing_function=self.smoothing().method1
        )
        bleu_scores.append(bleu)

      return np.mean(bleu_scores)

  def quality_score(self, texts):
    '''Compute the average quality score for a list of texts'''
    scores = [self.quality_filter.calculate_quality_score(text) for text in texts]
    return {'mean': np.mean(scores),
            'scores': scores}

  def load_training_stories(self, seed=42):
      '''Load a sample of training stories for novelty computation'''
      logging.info(f'Loading {self.training_samples} training stories for novelty computation')

      full_dataset = load_dataset("skeskinen/TinyStories-GPT4", split="train")
      full_dataset = full_dataset.remove_columns(
            [c for c in full_dataset.column_names if c not in ['story', 'features']])
      assert len(full_dataset) == 2745100

      splits = full_dataset.train_test_split(test_size=10000, seed=42, shuffle=True)

      train_dataset = splits ["train"] #Not needed for inference
      #test_dataset_raw = splits["test"]

      assert len(train_dataset) == 2735100
      #assert len(test_dataset_raw) == 10000

      random.seed(seed)
      sample_indices = random.sample(range(len(train_dataset)), self.training_samples)

      self.training_stories = [train_dataset[i]['story'] for i in sample_indices]


  def novelty_score(self, texts, max_n=4):
    '''Compute the novelty score by measuring how different generated texts are from training data'''

    self.load_training_stories()

    training_samples = random.sample(self.training_stories, self.training_samples)
    training_tokens = [self.preprocess_text(sample) for sample in training_samples]
    novelty_scores = []

    for text in texts:
      gen_tokens = self.preprocess_text(text)
      #Compute bleu score against training references
      bleu_score = sentence_bleu(
          training_tokens,
          gen_tokens,
          weights=[1.0/max_n] * max_n,
          smoothing_function=self.smoothing().method1
      )

      #Convert bleu to novelty:
      novelty = 1.0 - bleu_score
      novelty_scores.append(novelty)

    return np.mean(novelty_scores)


  def compute_all_metrics(self, texts):
      metrics = {
          'self_bleu': self.self_bleu(texts),
          'quality_score': self.quality_score(texts),
          'novelty_score': self.novelty_score(texts)
      }
      return metrics


class StoryInference:
  '''Handles loading the LoRA model and performing inference'''

  def __init__(self, model_path, base_model_name=None, lora_rank=16):
    '''
    model_path: path to the trained model weights
    base_model_name: name of the model used
    '''
    self.drive_folder_id = model_path
    self.base_model_name = base_model_name
    self.lora_rank = lora_rank
    self.tokenizer = None
    self.model = None
    self.weights_dir = '/content/weights'
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model_path = os.path.join(self.weights_dir, f'lora-R{lora_rank}')

  def download_weights(self):
    '''Download weights from drive folder with gdown'''
    logger.info(f'Downloading weights from drive folder {self.drive_folder_id}')

    os.makedirs(self.weights_dir, exist_ok=True)

    folder_url = f'https://drive.google.com/drive/folders/{self.drive_folder_id}'
    gdown.download_folder(folder_url, output=self.weights_dir, quiet=False, use_cookies=False)

    #Find the correct LoRA rank folder
    downloaded_folders = [d for d in os.listdir(self.weights_dir) if os.path.isdir(os.path.join(self.weights_dir, d))]

    #Look for the folder with the correct rank
    target_folder = None
    for folder in downloaded_folders:
          if f'lora-R{self.lora_rank}' in folder or f'rank-{self.lora_rank}' in folder or f'r{self.lora_rank}' in folder.lower():
              target_folder = folder
              break

    if target_folder:
            old_path = os.path.join(self.weights_dir, target_folder)
            # Move to expected path
            if old_path != self.model_path:
                if os.path.exists(self.model_path):
                    shutil.rmtree(self.model_path)
                shutil.move(old_path, self.model_path)
            logger.info(f'Using LoRA weights from: {self.model_path}')
    else:
            raise FileNotFoundError(f"Could not find LoRA weights for rank {self.lora_rank}")


  def load_model(self):
    '''Load the model and the tokenizer'''

    logger.info(f'Loading LoRA model with rank {self.lora_rank}')

    if not os.path.exists(self.model_path):
      self.download_weights()

    #Load tokenizer:
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    #Load LoRA:
    peft_config = PeftConfig.from_pretrained(self.model_path)
    self.base_model_name = peft_config.base_model_name_or_path

    logger.info(f'Using the base model {self.base_model_name}')

    #Load the base model:
    base_model = AutoModelForCausalLM.from_pretrained(
        self.base_model_name,
        torch_dtype=torch.float16, #accelerate inference
        device_map='auto'
    )

    #load LoRA weights:
    self.model = PeftModel.from_pretrained(base_model, self.model_path)
    self.model.eval()

    logger.info('Model loaded')
    return self.model, self.tokenizer

  def compute_loss(self, texts):
    '''Compute avg cross entropy loss, perplexity'''

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

        # Tokenize the full text (Byte Pair Encoding based tokenizer)
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
            input_ids = inputs['input_ids'].squeeze(0)  #Remove batch dimension
            labels = torch.full_like(input_ids, -100)  #Initialize all as -100

            #Only compute loss on story tokens
            if prompt_length < len(input_ids):
                labels[prompt_length:] = input_ids[prompt_length:] #Labels are token ids of the gt tokens

            #Compute cross entropy loss
            shift_logits = logits[..., :-1, :].contiguous() #The model predicts the next token given the current one
            shift_labels = labels[1:].contiguous()

            #Only compute loss where labels != -100
            valid_mask = (shift_labels != -100)

            if valid_mask.sum() > 0:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                #Only average over valid tokens
                valid_losses = losses[shift_labels.view(-1) != -100]
                avg_loss = valid_losses.mean().item()
                perplexity = np.exp(avg_loss)
            else:
                avg_loss = 0.0
                perplexity = 1.0

        batch_losses.append(avg_loss)
        batch_perpl.append(perplexity)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return batch_losses, batch_perpl

  def generate_story(self, features=None, max_length=300, temperature=0.8, top_p=0.9, do_sample=True, num_return_sequences=1):
      '''Generate a story given features'''

      # Create prompt
      if features and len(features) > 0:
          if isinstance(features, list):
              features_str = ", ".join(features)
          else:
              features_str = str(features)
          prompt = f'Features: {features_str}\nStory: '
      else:
          prompt = 'Story: '

      # Tokenize prompt
      inputs = self.tokenizer(
          prompt,
          return_tensors='pt',
          padding=False
      ).to(self.device)

      # Set up generation config
      generation_config = GenerationConfig(
          max_length=max_length,
          temperature=temperature, #Control the randomness of the generation
          top_p=top_p, #Ignore very unlikely tokens
          do_sample=do_sample, #Enables probabilistic selection rather than selecting the most likely token
          num_return_sequences=num_return_sequences,
          pad_token_id=self.tokenizer.eos_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
          repetition_penalty=1.1,
          length_penalty=1.0
      )

      with torch.no_grad():
          # Generate
          outputs = self.model.generate(
              **inputs,
              generation_config=generation_config
          )

          # Decode generated text
          generated_stories = []
          for output in outputs:
              full_text = self.tokenizer.decode(output, skip_special_tokens=True)

              # Extract just the story part
              if 'Story: ' in full_text:
                  story = full_text.split('Story: ', 1)[1].strip()
              else:
                  story = full_text.strip()

              generated_stories.append(story)

      # Clear GPU cache
      if torch.cuda.is_available():
          torch.cuda.empty_cache()

      return generated_stories if num_return_sequences > 1 else generated_stories[0]

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

    #train_dataset = splits ["train"] #Not needed for inference
    test_dataset_raw = splits["test"]

    #assert len(train_dataset) == 2735100
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

    #Exclude too short or too long stories (that the model cannot process)
    filtered_dataset = preprocessed_dataset.filter(
          lambda x: self.min_length <= x['token_length'] <= self.max_tokens
      )

    test_dataset = filtered_dataset

    logger.info(f'Test dataset size: {len(test_dataset)}')
    return test_dataset

def evaluate_model(inference_model, test_dataset, output_file=None, batch_size=8):
  '''Model evaluation using batched processing'''

  logger.info('Starting model inference')
  results = []
  all_losses = []
  all_perpl = []

  for i in tqdm(range(0, len(test_dataset), batch_size), desc='Evaluating batches'):
    batch_end = min(i + batch_size, len(test_dataset))
    batch_data = test_dataset.select(range(i, batch_end))

    #Prepare batch data:
    batch_features = []
    batch_stories = []
    batch_full_texts = []

    for example in batch_data:

      raw_features = example['features'] if example['features'] else []
      features = [feature.strip().lower() for feature in raw_features]
      original_story = example['story']

      #Create the same format used in training
      if isinstance(features, list) and len(features) > 0:
          features_str = ", ".join(features)
          prompt = f'Features: {features_str}\nStory: '
      else:
          prompt = 'Story: '

      full_text = prompt + original_story

      batch_features.append(features)
      batch_stories.append(original_story)
      batch_full_texts.append(full_text)

    #Compute losses for the batch
    batch_losses, batch_perpl = inference_model.compute_loss(batch_full_texts)
    all_losses.extend(batch_losses)
    all_perpl.extend(batch_perpl)

    #Store results
    for j, (features, story, loss, perplexity) in enumerate(zip(batch_features, batch_stories, batch_losses, batch_perpl)):
        results.append({
            'features': features,
            'story': story,
            'loss': loss,
            'perplexity': perplexity
        })

  #Calculate average loss
  avg_loss = np.mean(all_losses) if all_losses else 0.0
  avg_perpl = np.mean(all_perpl) if all_perpl else 1.0

  logger.info(f'Average loss: {avg_loss:.4f}')
  logger.info(f'Average perplexity: {avg_perpl:.4f}')

  return results

def demonstrate_generation(inference_model, num_examples=3):
    '''Story generation with different feature'''

    example_features = [
        ['dialogue'],
        ['twist'],
        ['moralvalue'],
        ['foreshadowing', 'conflict'],
    ]

    print("\n" + "="*60)
    print("GENERATED STORIES DEMONSTRATION")
    print("="*60 + "\n")

    for i, features in enumerate(example_features[:num_examples], 1):
        print(f"--- Example {i} ---")
        print(f"Features: {features}")
        print()

        # Generate story
        story = inference_model.generate_story(
            features=features,
            max_length=200,
            temperature=0.8,
            top_p=0.9
        )

        print(f"Generated Story:")
        print(story)
        print("\n" + "-"*50 + "\n")

def metrics_eval(inference_model, test_dataset, num_samples=100, seed=42):
    '''evaluation function for the metrics on a smaller test set'''
    print("Generating stories for metric computation")

    np.random.seed(seed)

    #Sample some examples for generation
    sample_indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    generated_stories = []
    original_stories = []


    for idx in sample_indices:
        example = test_dataset[int(idx)]
        features = example['features'] if example['features'] else []
        original_story = example['story']

        #Generate story
        story = inference_model.generate_story(
            features=features,
            max_length=200,
            temperature=0.8,
            top_p=0.9
        )
        generated_stories.append(story)
        original_stories.append(original_story) #Used to compute the quality scores of the original stories and confront it with the generated ones

    #Metrics computation
    evaluator = EvaluationMetrics()
    metrics = evaluator.compute_all_metrics(generated_stories)

    #quality score for original stories:
    original_quality = evaluator.quality_score(original_stories)

    return metrics, original_quality

def main():

    #Test parameter definition:
    seed = 42
    model_path = '1G5JOnE9Aw4mqtiWfQyMdNCQpQawfYa6C'  #Drive folder ID
    base_model = 'distilgpt2'
    batch_size = 8
    lora_rank = 16 #Select from 8, 16 or 32 as lora rank
    #Set random seed:
    torch.manual_seed(seed)

    #Initialize inference model:
    inference_model = StoryInference(model_path, base_model, lora_rank)
    model, tokenizer = inference_model.load_model()

    logger.info('Running story generation ')

    #story generation
    demonstrate_generation(inference_model, num_examples=3)

    logger.info('Running evaluation')

    #Load and process test dataset
    processor = TestDataProcessor(tokenizer)
    test_dataset = processor.load_test_dataset()

    #Metric evaluation on a subset of the test set
    logger.info('Metrics computation')
    metrics, original_quality = metrics_eval(inference_model, test_dataset, num_samples=25, seed=seed)

    logger.info(f"Self_BLEU score: {metrics['self_bleu']}")
    logger.info(f"Novelty score: {metrics['novelty_score']}")

    logger.info(f"Generated stories quality score: {metrics['quality_score']['mean']}")
    logger.info(f"Original stories quality score: {original_quality['mean']}")

    # Run evaluation
    results = evaluate_model(
        inference_model,
        test_dataset,
        batch_size=batch_size
    )


    logger.info('Evaluation completed.')

if __name__ == "__main__":
    main()