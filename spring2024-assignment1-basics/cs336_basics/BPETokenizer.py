from tokenizers.tokenizers import Tokenizer, decoders
from tokenizers import models, pre_tokenizers


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = {v.decode("utf-8"): k for k, v in vocab.items()}
        self.merges = [(a.decode("utf-8"), b.decode("utf-8")) for a, b in merges]
        self.special_tokens = special_tokens or []

        bpe_model = models.BPE(vocab=self.vocab, merges=self.merges, unk_token="<unk>")
        self.tokenizer = Tokenizer(bpe_model)

        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        self.tokenizer.decoder = decoders.BPEDecoder()


    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> bytes:
        return self.tokenizer.decode(ids)

