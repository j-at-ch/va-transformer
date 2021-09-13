class Mappings:
    def __init__(self,
                 mappings,
                 pad_token=None,
                 sos_token=None,
                 eos_token=None,
                 pad_guide_token=None,
                 sos_guide_token=None,
                 eos_guide_token=None
                 ):
        self.itemid2token = mappings['itemid2token']
        self.token2itemid = mappings['token2itemid']
        self.token2trcount = mappings['token2trcount']
        self.gn2gt = {  # todo incorporate into preprocessing pipeline
            'XLOW': 1, 'LOW': 2, 'MID': 3, 'HIGH': 4, 'XHIGH': 5, 'CAT': 6
        }
        self.gt2gn = {v: k for k, v in self.gn2gt.items()}
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_guide_token = pad_guide_token
        self.sos_guide_token = sos_guide_token
        self.eos_guide_token = eos_guide_token

        if pad_token is not None:
            self.append_special_(self.itemid2token, self.token2itemid, '[PAD]', pad_token)
        if sos_token is not None:
            self.append_special_(self.itemid2token, self.token2itemid, '[SOS]', sos_token)
        if eos_token is not None:
            self.append_special_(self.itemid2token, self.token2itemid, '[EOS]', eos_token)

        if pad_guide_token is not None:
            self.append_special_(self.gn2gt, self.gt2gn, '[PAD]', pad_guide_token)
        if sos_guide_token is not None:
            self.append_special_(self.gn2gt, self.gt2gn, '[SOS]', sos_guide_token)
        if eos_guide_token is not None:
            self.append_special_(self.gn2gt, self.gt2gn, '[EOS]', eos_guide_token)

        self.num_tokens = len(self.itemid2token)
        self.num_guide_tokens = len(self.gn2gt)

    @staticmethod
    def append_special_(n2t, t2n, name, token):
        n2t[name] = token
        t2n[token] = name

    def top_n_train_tokens(self, n):
        d = sorted(self.token2trcount.items(), key=lambda item: item[1], reverse=True)
        return dict(d[0:n])

    def decode_token(self, token):
        return str(self.token2itemid[token])

    def decode_tokens(self, tokens):
        return ' '.join(list(map(self.decode_token, tokens)))


class Labellers:
    def __init__(self, mappings, d_items_df):
        self.mappings = mappings
        self.d_items_df = d_items_df

    def token2label(self, token):
        if token == self.mappings.pad_token:
            return '[PAD]'
        elif token == self.mappings.sos_token:
            return '[SOS]'
        elif token == self.mappings.eos_token:
            return '[EOS]'
        else:
            itemid = self.mappings.token2itemid[token]
            x = self.d_items_df.loc[itemid, 'LABEL']
        return x

    def tokens2label_string(self, tokens):
        return '\n\t -> '.join(list(map(self.token2label, tokens)))