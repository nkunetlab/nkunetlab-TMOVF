# nkunetlab-TMOVF

## Code framework

- `train_source_model`: The training code for 8 source models and 4 unrelated models from googlenet.
- `finetune`: The fine-tuning code for each of the 8 source models.
- `knowledge_distilltion`: The knowledge-distilltion code for each of the 8 source models which distilled to googlenet.
- `prune`: Generic pruning code.
- `dataset`: The code for reading the dataset.
  - WARNING: `new_imagenet.py` can also be used to read the style migration dataset and steganography dataset of mini-imagenet.
- `get_outliers`: The code that calls the outlier algorithm, the tested objects are tested according to different script files: 
  - the test set of all models using the corresponding dataset
  - the train set of all models using the corresponding dataset
  - the test set of mini-imagenet corresponding models using steganographic datasets/style migration dataset
  - the train set of mini-imagenet corresponding models using steganographic datasets/style migration dataset
  - the train set of all models using part of the train set of the unrelated dataset cars
