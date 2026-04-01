import re

def create_vocab(text):
    text_preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    text_preprocessed = [item.strip() for item in text_preprocessed if item.strip()]
    all_words = sorted(set(text_preprocessed))
    all_words.extend(["[UNK]", "[EOS]"])
    vocab = {string:integer for integer, string in enumerate(all_words)}
    return vocab

class myTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        text_preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        text_preprocessed = [item.strip()  for item in text_preprocessed if item.strip()]
        text_preprocessed = [item if item in self.str_to_int else "[UNK]" for item in text_preprocessed]
        
        idx = [self.str_to_int[s] for s in text_preprocessed]
        return idx
    
    def decode(self, idx):
        text = " ".join([self.int_to_str[i] for i in idx])
        text = re.sub(r'\s+([;,.?!"()\'])', r'\1', text)
        return text