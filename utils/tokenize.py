# from keras_bert import Tokenizer
import codecs
import unicodedata
from transformers import BertTokenizer

# class HBTokenizer(Tokenizer):
#     def _tokenize(self, text):
#         if not self._cased:
#             text = unicodedata.normalize('NFD', text)
#             text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
#             text = text.lower()
#         spaced = ''
#         for ch in text:
#             if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
#                 continue
#             else:
#                 spaced += ch
#         tokens = []
#         for word in spaced.strip().split():
#             tokens += self._word_piece_tokenize(word)
#             tokens.append('[unused1]')  # [unused*]这些标记是未经训练的（随即初始化），是Bert预留出来用来增量添加词汇的标记，所以我们可以用它们来指代任何新字符。
#         return tokens

# def get_tokenizer(vocab_path):
#     token_dict = {}
#     with codecs.open(vocab_path, 'r', 'utf8') as reader:
#         for line in reader:
#             token = line.strip()
#             token_dict[token] = len(token_dict)
#     return HBTokenizer(token_dict, cased=True)

class NBTokenizer(BertTokenizer):       # New Bert Tokenizer
    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
                split_tokens.append('[unused1]')
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

def get_tokenizer(vocab_path):
    return NBTokenizer(vocab_path, do_lower_case=False)

