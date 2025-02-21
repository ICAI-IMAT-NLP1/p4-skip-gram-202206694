import torch
from data_processing import load_and_preprocess_data, create_lookup_tables, subsample_words

file_path = "data/text8"

print("Loading data...")
tokens = load_and_preprocess_data(file_path)
print(f"Tokens loaded: {len(tokens)}")

print("Creating lookup tables...")
vocab_to_int, int_to_vocab = create_lookup_tables(tokens)
print(f"Vocabulary size: {len(vocab_to_int)}")

print("Starting subsampling...")
train_words, freqs = subsample_words(tokens, vocab_to_int)
print(f"Subsampling done. Final word count: {len(train_words)}")
