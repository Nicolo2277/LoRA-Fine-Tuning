## Low Rank Adaptation Fine-Tuning for story generation

**Overview**
This project implements a story generation pipeline using Low-Rank Adaptation (LoRA) fine-tuning on the distilgpt2 model from Hugging Face. Given six input features describing story attributes (e.g. 'dialogue' or 'twist'), our model generates coherent narratives. We evaluate generated stories using multiple metrics:
- **Quality Score**:  A custom score to assess overall story quality, applied during both inference and preprocessing.
- **Self-Bleu**: Measures fluency and diversity by comparing generated stories against each other.
- **Novelty Score**: Quantifies generated story uniqueness relative to the training set.
- **Perplexity & Average Cross-Entropy**: Standard language model loss metrics.

For detailed methodology on training and results, see the [project report](docs/report.pdf).


**Inference Colab Notebook**
We provide a ready-to-run Colab notebook for generating stories with the fine-tuned LoRA adapter, including the weights for different LoRA ranks:

- **Colab Notebook**: https://colab.research.google.com/drive/1QMm6N1Wn1-Pc60Zt-MX3qhYCBCpG-dMr#scrollTo=XOfCMScuXjvU

