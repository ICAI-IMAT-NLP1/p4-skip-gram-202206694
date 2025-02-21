from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file

    # Preprocess and tokenize the text
    # TODO
    tokens: List[str] = tokenize(text)

    return tokens


def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    word_counts: Counter = Counter(words)

    # Sorting the words from most to least frequent in text occurrence.
    sorted_vocab: List[str] = [word for word, _ in word_counts.most_common()]

    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = {i: word for i, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: i for i, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, reducing 
    the presence of frequent words according to Mikolov's subsampling technique.
    """

    print("Converting words to integers...")
    int_words = torch.tensor([vocab_to_int[word] for word in words], dtype=torch.long)

    print("Calculating frequencies...")
    total_count = len(int_words)
    word_counts = torch.bincount(int_words, minlength=len(vocab_to_int))

    # Convert to probability distribution
    freqs = word_counts.float() / total_count

    # Compute subsampling probability (avoid division by zero)
    probs = 1 - torch.sqrt(threshold / (freqs + 1e-10))

    print("Applying subsampling...")
    keep_prob = torch.rand(len(int_words))
    train_words = int_words[keep_prob > probs[int_words]]

    print(f"Subsampled words: {len(train_words)}")
    return train_words.tolist(), {word: freqs[i].item() for word, i in vocab_to_int.items()}





def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    # TODO
    target_words: List[str] = []

    R = torch.randint(1, window_size + 1, (1,)).item()
    start = max(0, idx - R)
    end = min(len(words), idx + R + 1)

    for i in range(start, end):
        if i != idx:
            target_words.append(words[i])

    return target_words


def get_batches(words: List[int], batch_size: int, window_size: int = 5) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """
    # TODO
    for idx in range(0, len(words), batch_size):
        inputs, targets = [], []
        batch = words[idx:idx + batch_size]

        for i in range(len(batch)):
            target_context = get_target(batch, i, window_size)
            inputs.extend([batch[i]] * len(target_context))
            targets.extend(target_context)

        yield inputs, targets


def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """
    # TODO
    valid_examples: torch.Tensor = torch.cat(
        (torch.randint(0, valid_window, (valid_size // 2,)),
         torch.randint(valid_window, valid_window * 2, (valid_size // 2,)))
    ).to(device)

    embed_vectors = embedding.weight
    embed_norm = embed_vectors / embed_vectors.norm(dim=1, keepdim=True)

    valid_vectors = embed_vectors[valid_examples]
    similarities: torch.Tensor = torch.mm(valid_vectors, embed_norm.t())

    return valid_examples, similarities
