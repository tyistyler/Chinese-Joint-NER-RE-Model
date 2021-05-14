from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
from utils.tokenize import get_tokenizer
import numpy as np
from random import choice

tokenizer = get_tokenizer('data/bert-base-cased/vocab.txt')

bert_max_seq_len = 512

def find_head_idx(source, target):  # 查找entity开头时会存在问题，当句子中出现多个相同entity时，只会选择第一个
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i+target_len] == target:
            return i
    return -1

class NYTDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):  # 当实例对象Object使用Object[key]的方式来取值时，就会调用__getitem__方法
        ins_json_data = self.json_data[idx]  # 数据集json串
        '''
            {
                "text": "But that spasm of irritation by a master intimidator was minor compared with what Bobby Fischer , the erratic former world chess champion , dished out in March at a news conference in Reykjavik , Iceland .",
                "triple_list": [
                    [
                        "Iceland",
                        "/location/country/capital",
                        "Reykjavik"
                    ],
                    [
                        "Iceland",
                        "/location/location/contains",
                        "Reykjavik"
                    ],
                    [
                        "Fischer",
                        "/people/person/nationality",
                        "Iceland"
                    ],
                    [
                        "Fischer",
                        "/people/deceased_person/place_of_death",
                        "Reykjavik"
                    ]
                ]
            }
        '''
        text = ins_json_data['text']
        text = ' '.join(text.split()[:self.config.max_seq_len])
        tokens = self.tokenizer.tokenize(text)                        # keras_bert.tokenizer
#         tokens_ = self.tokenizer.tokenize(text)                         # BertTokenizer       二者的确有不同，本行没有[CLS]和[SEP]
#         tokens = [self.tokenizer.cls_token] + tokens_ + [self.tokenizer.sep_token]
        if len(tokens) > bert_max_seq_len:
            tokens = tokens[:bert_max_seq_len]
        text_len = len(tokens)
        if not self.is_test:
            s2ro_map = {}  # subject to relation and object
            for triple in ins_json_data['triple_list']:  #
                triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1],self.tokenizer.tokenize(triple[2])[1:-1])  # keras_bert.tokenizer---[1:-1]的作用是去掉开头的[CLS]跟结尾的[SEP]
#                 triple = (self.tokenizer.tokenize(triple[0]), triple[1], self.tokenizer.tokenize(triple[2]))

                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))
            if s2ro_map:
                token_ids, segment_ids = self.tokenizer.encode(first=text)  # keras_bert.tokenizer
#                 token_ids = self.tokenizer.encode(text=text, add_special_tokens=True)
#                 segment_ids = [0 for _ in range(len(token_ids))]
                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                token_ids = np.array(token_ids)
                masks = np.array(masks) + 1
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))  # list中随机选择一个数
                '''
                    随机选择一个subject参与训练而不是所有的subject都参与训练
                    原因：因为模型采用了级联的two-step解码，所以
                    such a setting(随机选择的操作) makes it easier to proceed with the latter relation-specific object tagging part
                '''
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1

                obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros(
                    (text_len, self.config.rel_num))
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
                       ins_json_data['triple_list'], tokens
            else:
                print(idx)
                print(tokens)
                print('-----------------')
                return None
        else:
            token_ids, segment_ids = self.tokenizer.encode(first=text)
#             token_ids = self.tokenizer.encode(text=text, add_special_tokens=True)
#             segment_ids = [0 for _ in range(len(token_ids))]
            masks = segment_ids
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            obj_heads, obj_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
                   ins_json_data['triple_list'], tokens


def nyt_collate_fn(batch):
    batch = list(filter(lambda x:x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)        # 按句子长度排序，从大到小
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj_heads = torch.Tensor(cur_batch, max_text_len, 24).zero_()
    batch_obj_tails = torch.Tensor(cur_batch, max_text_len, 24).zero_()

    for i in range(cur_batch):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
        batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
        batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
        batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
        batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
        batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'sub_heads': batch_sub_heads,
            'sub_tails': batch_sub_tails,
            'sub_head': batch_sub_head,
            'sub_tail': batch_sub_tail,
            'obj_heads': batch_obj_heads,
            'obj_tails': batch_obj_tails,
            'triples': triples,
            'tokens': tokens}




def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=nyt_collate_fn):
    dataset = NYTDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
        '''
            pin_memory: 锁页内存，一般来说，在GPU训练的时候设置成True，在CPU上设置成False
            collate_fn: 将Dataset中的单个数据拼成batch的数据
        '''
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data




