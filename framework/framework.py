import torch.optim as optim
from torch import nn
import os
import data_loader
import torch.nn.functional as F
import torch
import numpy as np
import json
import time

class Framework(object):
    def __init__(self, con):
        self.config = con

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
        # initialize the model
        ori_model = model_pattern(self.config)
        ori_model.to(self.device)

        # define the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr=self.config.learning_rate) # 在网络中固定部分参数进行训练
        # optimozer = optim.Adam(ori_model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.99))

        # whether use multi GPU
        if self.config.multi_gpu:
            model = nn.DataParrllel(ori_model)
        else:
            model = ori_model

        # define the loss function
        def loss(gold, pred, mask):
            '''
            :param gold:    [batch_size, seq_len]
            :param pred:    [batch_size, seq_len, 1]
            :param mask:    [batch_size, seq_len]
            :return:
            '''
            pred = pred.squeeze(-1)                                         # [batch_size, seq_len]
            los = F.binary_cross_entropy(pred, gold, reduction='none')      # [batch_size, seq_len]
            if los.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            los = torch.sum(los * mask) / torch.sum(mask)                   # los * mask会把mask=0的位置置0---torch.tensor(0.8204)
            return los

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        # get the data loader
        train_data_loader = data_loader.get_loader(self.config, prefix=self.config.train_triples)
        dev_data_loader = data_loader.get_loader(self.config, prefix=self.config.dev_triples, is_test=True)

        # other
        model.train()
        global_step = 0
        loss_sum = 0

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.n_epoch):
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()
            while data is not None:
                pre_sub_heads, pre_sub_tails, pre_obj_heads, pre_obj_tails = model(data)

                sub_heads_loss = loss(data['sub_heads'], pre_sub_heads, data['mask'])
                sub_tails_loss = loss(data['sub_tails'], pre_sub_tails, data['mask'])
                obj_heads_loss = loss(data['obj_heads'], pre_obj_heads, data['mask'])
                obj_tails_loss = loss(data['obj_tails'], pre_obj_tails, data['mask'])

                total_loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

                optimizer.zero_grad()
                total_loss.backward()
                # 更新参数
                optimizer.step()
                # 可以补充一个，更新学习率
                # scheduler.step()


                global_step += 1
                loss_sum += total_loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, setp: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                                 format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))
                    loss_sum = 0
                    start_time = time.time()

                data = train_data_prefetcher.next()

            if (epoch + 1) % self.config.test_epoch == 0:
                eval_start_time = time.time()
                model.eval()
                # call the thest function
                precision, recall, f1_score = self.test(dev_data_loader, model)
                model.train()
                self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.
                             format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_epoch = epoch
                    best_precision = precision
                    best_recall = recall
                    self.logging("saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
                                 format(best_epoch, best_f1_score, precision, recall))
                    # save the best model
                    path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                    torch.save(model.state_dict(), path)
            # manully release teh unused cache
            torch.cuda.empty_cache()

        self.logging("finish training")
        self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
                     format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))

    def test(self, test_data_loader, model, output=False, h_bar=0.5, t_bar=0.5):
        if output:
            # check the result dir
            if not os.path.exists(self.config.result_dir):
                os.mkdir(self.config.result_dir)
            path = os.path.join(self.config.result_dir, self.config.result_save_name)

            fw = open(path, 'w')

        orders = ['subject', 'relation', 'object']

        def to_tup(triple_list):
            ret = []
            for triple in triple_list:
                ret.append(tuple(triple))
            return ret

        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        id2rel = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[0]
        correct_num, predict_num, gold_num = 0, 0, 0

        while data is not None:
            with torch.no_grad():
                token_ids = data['token_ids']
                tokens = data['tokens'][0]          # batch_size=1, 此时data['tokens']是一个二维列表
                mask = data['mask']
                encoded_text = model.get_encoded_text(token_ids, mask)
                pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
                sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], np.where(pred_sub_tails.cpu()[0] > t_bar)[0]
                '''
                    pred_sub_heads.cpu()                --[1, 81, 1]
                    pred_sub_heads.cpu()[0]             --[81, 1]
                    np.where(pred_sub_heads.cpu()[0]    --返回满足条件的索引
                    
                    
                '''
                subjects = []
                for sub_head in sub_heads:
                    sub_tail = sub_tails[sub_tails >= sub_head]
                    if len(sub_tail) > 0:
                        sub_tail = sub_tail[0]                                      # 最近匹配原则
                        subject = tokens[sub_head: sub_tail]
                        subjects.append((subject, sub_head, sub_tail))              # 记录当前句子所有的subject
                if subject:
                    triple_list = []
                    # [subject_num, seq_len, bert_dim]
                    repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                    # [subject_num, 1, seq_len]
                    sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                    sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()

                    for subject_idx, subject in enumerate(subjects):                # 遍历每一个subject
                        sub_head_mapping[subject_idx][0][subject[1]] = 1            # 记录每个subject的head和tail
                        sub_head_mapping[subject_idx][0][subject[2]] = 1
                    sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)   # 数据没有发生变化，只是被转移到了GPU上
                    sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)

                    pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, repeated_encoded_text)
                    '''
                        pred_obj_heads  ---[subject_num, seq_len, rel_num]
                        pred_obj_tails  ---[subject_num, seq_len, rel_num]
                    '''
                    for subject_idx, subject in enumerate(subjects):
                        sub = subject[0]
                        sub = ''.join([i.lstrip('##') for i in sub])
                        sub = ' '.join(sub.split('[unused1]'))
                        obj_heads, obj_tails = np.where(pred_obj_heads.cpu()[subject_idx] > h_bar), np.where(pred_obj_tails.cpu()[subject_idx] > t_bar)
                        '''
                            pred_obj_heads.cpu()[subject_idx]   ---[seq_len, rel_num]
                        '''
                        for obj_head, rel_head in zip(*obj_heads):
                            for obj_tail, rel_tail in zip(*obj_tails):
                                if obj_head <= obj_tail and rel_head == rel_tail:
                                    rel = id2rel[str(int(rel_head))]
                                    obj = tokens[obj_head: obj_tail]
                                    obj = ''.join([i.lstrip('##') for i in obj])
                                    obj = ' '.join(obj.split('[unused1]'))
                                    triple_list.append((sub, rel, obj))
                                    break
                        triple_set = set()
                        for s, r, o in triple_list:
                            triple_set.add((s, r, o))
                        pred_list = list(triple_set)
                else:
                    pred_list = []

                pred_triples = set(pred_list)
                gold_triples = set(to_tup(data['triples'][0]))      #   由于batch_size=1, 所以data['triples']是个三维矩阵

                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                if output:
                    result = json.dumps({
                        'triple_list_gold':[
                            dict(zip(orders, triple)) for triple in gold_triples
                        ],
                        'triple_list_pred': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ],
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                        ],
                        'lack': [
                            dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                        ]

                    }, ensure_ascii=False)
                    fw.write(result + '\n')

                data = test_data_prefetcher.next()

        print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, f1_score

    def testall(self, model_pattern):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_triples, is_test=True)
        precision, recall, f1_score = self.test(test_data_loader, model, output=True)
        print("f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}".format(f1_score, precision, recall))

