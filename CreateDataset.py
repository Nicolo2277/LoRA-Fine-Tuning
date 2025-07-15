###########################################################################################################
#Script for creating a balanced training set
'''
Dataset sampler to extraxt a subset of the original training set for the LoRA fine tuning task while mantaining balanced labels
'''

import logging
from pathlib import Path
from collections import Counter, defaultdict
import random
import hashlib
import numpy as np
from datasets import load_dataset, Dataset
import re

#Logging setup:
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class SamplingConfig:
    '''Configuration for sampling strategy'''
    target_samples = 10000
    min_story_length = 50  # minimum words
    max_story_length = 1000  # maximum words
    min_features_per_story = 1
    balance_features = True
    quality_threshold = 0.7  # story quality score threshold
    seed = 42

class StoryQualityFilter:
    '''Filters stories based on quality metrics'''

    def __init__(self):
        # Common bad patterns to filter out
        self.bad_patterns = [
            r'\b(lorem ipsum|placeholder|test|example)\b',
            r'^(.{1,10})\1{3,}',  # repeated short patterns
            r'[^\w\s]{10,}',  # too many special characters
            r'\b\w{50,}\b',  # extremely long words
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.bad_patterns]

    def calculate_quality_score(self, story):
        '''Calculate quality score (0-1) for a story'''
        if not story or len(story.strip()) < 10:
            return 0.0

        score = 1.0

        # Decrease the score if we encounter bad patterns
        for pattern in self.compiled_patterns:
            if pattern.search(story):
                score -= 0.3

        words = story.split()
        sentences = story.split('.')

        # Penalize very short or very long stories
        word_count = len(words)
        if word_count < 20:
            score -= 0.4
        elif word_count > 2000:
            score -= 0.2

        # Check sentence variety
        if len(sentences) > 1:
            avg_sentence_length = word_count / len(sentences)
            if 5 <= avg_sentence_length <= 30:
                score += 0.1

        # Check vocabulary diversity
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        vocab_ratio = unique_words / max(word_count, 1)
        if vocab_ratio > 0.3:
            score += 0.1
        elif vocab_ratio < 0.1:
            score -= 0.2

        # Plus if dialogue is present
        if '"' in story or "'" in story:
            score += 0.1

        return max(0.0, min(1.0, score))

class FeatureBalancer:
    '''Handles feature extraction and balancing'''

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.feature_counts = Counter()
        self.feature_targets = {}

    def extract_and_clean_features(self, features_raw):
        '''Extract and clean features from raw feature data'''
        if not features_raw:
            return []

        # Clean features
        cleaned_features = []
        for feature in features_raw:
            #print(feature)

            feature = feature.strip().lower()

            cleaned_features.append(feature)

        return cleaned_features

    def analyze_feature_distribution(self, dataset):
        '''Analyze feature distribution in the dataset'''
        logger.info('Analyzing feature distribution')

        feature_counter = Counter()
        total_stories = 0

        for item in dataset:
            features = self.extract_and_clean_features(item.get('features', []))
            for feature in features:
                feature_counter[feature] += 1
            total_stories += 1

            if total_stories % 10000 == 0:
                logger.info(f'Analyzed {total_stories} stories')

        # Log top features
        top_features = feature_counter.most_common(20)
        logger.info("Top features:")
        for feature, count in top_features:
            logger.info(f"  {feature}: {count}")

        return dict(feature_counter)

    def calculate_feature_targets(self, feature_distribution, target_samples):
        """Calculate target counts for each feature to achieve balance"""
        total_feature_instances = sum(feature_distribution.values())

        # Filter out very rare features (appear less than 10 times)
        min_count = max(10, total_feature_instances // (target_samples * 10))
        filtered_features = {k: v for k, v in feature_distribution.items() if v >= min_count}

        # Calculate targets so that we can create a balanced subset of the original dataset
        num_features = len(filtered_features)
        avg_stories_per_feature = target_samples / num_features

        # Adjust targets based on original distribution but cap extremes
        targets = {}
        for feature, original_count in filtered_features.items():
            # Use geometric mean between equal distribution and original distribution
            equal_target = avg_stories_per_feature
            proportional_target = (original_count / total_feature_instances) * target_samples

            # Blend the two approaches
            target = int(np.sqrt(equal_target * proportional_target))

            min_target = max(10, target_samples // (num_features * 5))
            max_target = target_samples // 5

            targets[feature] = max(min_target, min(target, max_target))

        logger.info(f"Feature targets calculated. Range: {min(targets.values())} - {max(targets.values())}")
        return targets

class SmartDatasetSampler:
    '''Main sampler class'''

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.quality_filter = StoryQualityFilter()
        self.feature_balancer = FeatureBalancer(config)
        random.seed(config.seed)
        np.random.seed(config.seed)

    def _create_cache_key(self, dataset_identifier):
        '''Create a cache to avoid computation from scratch when rerunning the code'''
        config_str = f"{dataset_identifier}_{self.config.target_samples}_{self.config.seed}"
        config_str += f"_{self.config.balance_features}_{self.config.quality_threshold}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def sample_dataset(self, dataset, dataset_name,
                      cache_dir = "./sampled_datasets"):

        # Setup cache
        cache_key = self._create_cache_key(f"{dataset_name}_{len(dataset)}")
        cache_path = Path(cache_dir) / f"{dataset_name}_{cache_key}"

        # Check cache first
        if cache_path.exists():
            logger.info(f'Loading cached sampled dataset from {cache_path}')
            try:
                return Dataset.load_from_disk(str(cache_path))
            except Exception as e:
                logger.warning(f'Failed to load cache: {e}')

        logger.info(f'Sampling from provided dataset with {len(dataset)} samples')

        # Sample the dataset
        sampled_data = self._stratified_sample_by_features(dataset)

        # Create final dataset
        final_dataset = self._create_dataset(sampled_data)

        # Cache the result
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        final_dataset.save_to_disk(str(cache_path))
        logger.info(f'Cached sampled dataset to {cache_path}')

        return final_dataset

    def load_and_sample_dataset(self, dataset_name, dataset_split = 'train',
                               cache_dir = './sampled_datasets'):
        '''Load and sample the dataset from the provided cache'''

        # Setup cache
        cache_key = self._create_cache_key(dataset_name)
        cache_path = Path(cache_dir) / f"{dataset_name.replace('/', '_')}_{cache_key}"

        # Check cache first
        if cache_path.exists():
            logger.info(f'Loading cached sampled dataset from {cache_path}')
            try:
                return Dataset.load_from_disk(str(cache_path))
            except Exception as e:
                logger.warning(f'Failed to load cache: {e}')

        # Load original dataset
        logger.info(f'Loading dataset: {dataset_name}')
        try:
            full_dataset = load_dataset(dataset_name, split=dataset_split)
        except Exception as e:
            logger.error(f'Failed to load dataset: {e}')
            raise

        logger.info(f'Original dataset size: {len(full_dataset)}')

        return self.sample_dataset(full_dataset, dataset_name.replace('/', '_'), cache_dir)

    def _stratified_sample_by_features(self, dataset):
        '''Stratified sampling to balance feature distribution and create a more representative train dataset'''
        logger.info('Performing stratified sampling')

        # Check features distributions
        feature_distribution = self.feature_balancer.analyze_feature_distribution(dataset)
        feature_targets = self.feature_balancer.calculate_feature_targets(
            feature_distribution, self.config.target_samples
        )

        # Group stories by their primary features
        feature_to_stories = defaultdict(list)

        logger.info('Grouping stories by features')
        for idx, item in enumerate(dataset):
            story = item.get('story', '')

            # Basic filtering (the story must have a min and a max number of words)
            if not self._passes_basic_filters(story):
                continue

            quality_score = self.quality_filter.calculate_quality_score(story)
            if quality_score < self.config.quality_threshold: #We keep only good stories
                continue

            features = self.feature_balancer.extract_and_clean_features(item.get('features', []))
            if len(features) < self.config.min_features_per_story: #At least 1 feature
                continue

            # Add to feature groups (story can belong to multiple groups)
            story_data = {
                'story': story,
                'features': features,
                'quality_score': quality_score,
                'idx': idx
            }

            for feature in features:
                if feature in feature_targets:
                    feature_to_stories[feature].append(story_data)

            if idx % 10000 == 0:
                logger.info(f'Processed {idx} stories for grouping')

        # Sample from each feature group
        sampled_stories = []
        used_indices = set()

        logger.info('Sampling from feature groups ')
        for feature, target_count in feature_targets.items():
            available_stories = [s for s in feature_to_stories[feature] if s['idx'] not in used_indices]

            if not available_stories:
                continue

            # Sort by quality score and sample top stories
            available_stories.sort(key=lambda x: x['quality_score'], reverse=True)

            # Take the best stories up to target count
            sampled_count = min(target_count, len(available_stories))
            selected_stories = available_stories[:sampled_count]

            for story_data in selected_stories:
                if story_data['idx'] not in used_indices:
                    sampled_stories.append(story_data)
                    used_indices.add(story_data['idx'])

            logger.info(f"Feature '{feature}': sampled {len(selected_stories)} stories")

        # Remove idx field and shuffle
        final_stories = []
        for story_data in sampled_stories:
            del story_data['idx']
            final_stories.append(story_data)

        random.shuffle(final_stories)

        # Truncate to target size if we have too many stories
        if len(final_stories) > self.config.target_samples:
            final_stories = final_stories[:self.config.target_samples]

        logger.info(f'Stratified sampling completed: {len(final_stories)} stories')
        self._log_feature_distribution(final_stories)

        return final_stories

    def _passes_basic_filters(self, story):
        '''Check if story passes initial filters'''
        if not story or not isinstance(story, str):
            return False

        story = story.strip()
        if len(story) < 20:  # Too short
            return False

        words = story.split()
        word_count = len(words)

        return (self.config.min_story_length <= word_count <= self.config.max_story_length)

    def _create_dataset(self, sampled_data):
        '''Create the dataset'''
        logger.info('Creating final dataset')

        stories = [item['story'] for item in sampled_data]
        features = [item['features'] for item in sampled_data]
        quality_scores = [item['quality_score'] for item in sampled_data]

        # Create dataset
        dataset = Dataset.from_dict({
            'story': stories,
            'features': features,
            'quality_score': quality_scores
        })

        logger.info(f'Final dataset created with {len(dataset)} samples')
        return dataset

    def _log_feature_distribution(self, sampled_data):
        '''Log feature distribution in sampled data'''
        feature_counter = Counter()
        for item in sampled_data:
            for feature in item['features']:
                feature_counter[feature] += 1

        logger.info('Feature distribution in sampled data:')
        for feature, count in feature_counter.most_common(15):
            logger.info(f"  {feature}: {count}")

# Convenience function
def sample_from_dataset(dataset, config = None,
                       dataset_name = 'skeskinen/TinyStories-GPT4',
                       cache_dir = './sampled_datasets'):

    if config is None:
        config = SamplingConfig()

    sampler = SmartDatasetSampler(config)
    return sampler.sample_dataset(dataset, dataset_name, cache_dir)


def main():
    #Loading the dataset as instructed and extracting a subset of 10000 samples from the training set:

    full_dataset = load_dataset("skeskinen/TinyStories-GPT4", split="train")
    full_dataset = full_dataset.remove_columns([c for c in full_dataset.column_names if c not in ["story", "features"]])

    splits = full_dataset.train_test_split(test_size=10000, seed=42, shuffle=True)
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    #sample the training dataset
    sampled_train_dataset = sample_from_dataset(
        dataset=train_dataset,
        dataset_name="tinystories_train_presplit",
        cache_dir="./sampled_datasets"
    )

    print(f'Original train dataset: {len(train_dataset)} samples')
    print(f'Sampled train dataset: {len(sampled_train_dataset)} samples')
    print(f'Test dataset: {len(test_dataset)} samples')

    #save the datasets for training and testing
    sampled_train_dataset.save_to_disk('./final_datasets/sampled_train')
    test_dataset.save_to_disk('./final_datasets/test')

    return sampled_train_dataset, test_dataset

if __name__ == '__main__':
    sampled_train, test = main()
