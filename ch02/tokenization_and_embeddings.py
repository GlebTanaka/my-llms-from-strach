import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import urllib.request
import tiktoken

class SimpleTokenizerV1:
    """
    A basic tokenizer that splits text on specified patterns and converts tokens to IDs.
    """
    def __init__(self, vocab):
        self.str_to_int = vocab  # Dictionary mapping tokens to IDs
        self.int_to_str = {i: s for s, i in vocab.items()}  # Reverse mapping: IDs to tokens

    def encode(self, text):
        """Convert text to token IDs"""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """Convert token IDs back to text"""
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """
    An improved tokenizer that handles unknown words using a special <|unk|> token.
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        # Store the '<|unk|>' token ID for easy access
        self.unk_token_id = vocab['<|unk|>']

    def encode(self, text, allowed_special=None):
        """
        Convert text to token IDs, handling unknown words.
        
        Args:
            text: The input text to tokenize
            allowed_special: Set of special tokens to allow (ignored in this implementation)
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Use get() with unk_token_id as default for unknown words
        ids = [self.str_to_int.get(s, self.unk_token_id) for s in preprocessed]
        return ids

    def decode(self, ids):
        """Convert token IDs back to text"""
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class GPTDatasetV1(Dataset):
    """
    Dataset for training language models with a sliding window approach.
    Creates input-target pairs where each target is the input shifted by one token.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """
    Create a DataLoader for training language models.
    
    Args:
        txt: Input text
        batch_size: Number of sequences per batch
        max_length: Maximum sequence length
        stride: Step size for sliding window
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader object
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def create_token_embeddings(vocab_size, embedding_dim):
    """
    Create an embedding layer for tokens.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors
        
    Returns:
        PyTorch Embedding layer
    """
    return nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim
    )


def build_vocabulary_from_text(text):
    """
    Build a vocabulary from text.
    
    Args:
        text: Input text
        
    Returns:
        token_to_id: Dictionary mapping tokens to IDs
    """
    # Tokenize the text
    pattern = r'([,.:;?_!"()\']|--|\s)'
    tokens = [token.strip() for token in re.split(pattern, text) if token.strip()]
    
    # Create vocabulary
    vocab = sorted(set(tokens))
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    
    # Add special tokens
    next_id = len(token_to_id)
    token_to_id['<|endoftext|>'] = next_id
    token_to_id['<|unk|>'] = next_id + 1
    
    return token_to_id


def load_text_file(file_path, url=None):
    """
    Load text from a file, downloading it if necessary.
    
    Args:
        file_path: Path to the text file
        url: URL to download the file from if it doesn't exist
        
    Returns:
        The text content of the file
    """
    if not os.path.exists(file_path) and url:
        urllib.request.urlretrieve(url, file_path)
        
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def demonstrate_sliding_window(tokenizer, text, context_size=4, start_pos=50):
    """
    Demonstrate how sliding window works for language modeling.
    
    Args:
        tokenizer: Tokenizer to use
        text: Input text
        context_size: Size of the context window
        start_pos: Starting position in the encoded text
    """
    # Convert the raw text into token IDs
    enc_text = tokenizer.encode(text)
    print(f"Total number of tokens in the text: {len(enc_text)}")
    
    # Take a sample starting from the specified position
    enc_sample = enc_text[start_pos:]
    
    # Create input (x) and target (y) sequences
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    
    print("\nDemonstrating basic input-target pair:")
    print(f"Input sequence (x):  {x}")
    print(f"Target sequence (y): {y}")
    
    print("\nDemonstrating progressive context building with token IDs:")
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(f"Context {i} tokens: {context} ----> Next token: {desired}")
    
    print("\nDemonstrating progressive context building with decoded text:")
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(f"Context: '{tokenizer.decode(context)}' ----> Next token: '{tokenizer.decode([desired])}'")


# Example usage
if __name__ == "__main__":
    # Load text
    text = load_text_file(
        'the-verdict.txt',
        'https://raw.githubusercontent.com/GlebTanaka/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt'
    )
    
    # Build vocabulary
    token_to_id = build_vocabulary_from_text(text)
    
    # Create tokenizer
    tokenizer = SimpleTokenizerV2(token_to_id)
    
    # Demonstrate tokenization
    sample_text = "Hello world! This is an example."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Demonstrate sliding window
    demonstrate_sliding_window(tokenizer, text)
    
    # Create token embeddings
    vocab_size = len(token_to_id)
    embedding_dim = 128
    embedding_layer = create_token_embeddings(vocab_size, embedding_dim)
    
    # Example of using embeddings
    token_ids = torch.tensor([[0, 2, 1, 3], [1, 1, 4, 2]])
    embedded_tokens = embedding_layer(token_ids)
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Embeddings shape: {embedded_tokens.shape}")
    
    # Create dataloader
    dataloader = create_dataloader_v1(
        text,
        batch_size=2,
        max_length=8,
        stride=4
    )
    
    # Show first batch
    for batch in dataloader:
        inputs, targets = batch
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch targets shape: {targets.shape}")
        break