from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import pickle
import os

import BPETokenizer


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    tokenizer.train_from_iterator(lines, trainer)

    vocab = tokenizer.model.get_vocab()
    id_to_token_bytes = {v: k.encode("utf-8") for k, v in vocab.items()}

    merges_str = tokenizer.model.get_merges()
    merges_bytes = [(a.encode("utf-8"), b.encode("utf-8")) for (a, b) in merges_str]

    return id_to_token_bytes, merges_bytes
