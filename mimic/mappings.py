
class Mappings:
    def __init__(self, tokens,
                 index_shift=5,
                 pad_index=None,
                 bos_index=None,
                 eos_index=None
                 ):
        self.tokens = tokens
        self.index_shift = index_shift
        self.indexes = range(index_shift, index_shift + len(tokens))
        self.pad_index = pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index

    def tok2idx(self):
        D = dict(zip(self.tokens, self.indexes))
        if self.pad_index is not None:
            D['[PAD]'] = self.pad_index
        if self.bos_index is not None:
            D['[BOS]'] = self.bos_index
        if self.pad_index is not None:
            D['[EOS]'] = self.eos_index
        return D

    def idx2tok(self):
        return {v: k for k, v in self.tok2idx().items()}

    def get_idx(self, tok):
        return self.tok2idx()[tok]

    def get_tok(self, idx):
        return self.idx2tok()[idx]

    def get_tok_sequence(self, idx_sequence):
        return ' '.join(list(map(self.get_tok, idx_sequence)))
