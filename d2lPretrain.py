import torch
from torch import nn
from d2l import torch as d2l
from d2lDataset import load_data_wiki_2
from models.BertModel import *
import sys 
import os
import psutil
import time

#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    p = psutil.Process()
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, Timer()
    #animator = Animator(xlabel='step', ylabel='loss',
    #                        xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        
        print(p.io_counters())
        prev_read = p.io_counters()[2]
        prev_write = p.io_counters()[3]
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            #animator.add(step + 1,
            #             (metric[0] / metric[3], metric[1] / metric[3]))
            now_read = p.io_counters()[2]
            now_write = p.io_counters()[3]
            print(f'MLM loss {metric[0] / metric[3]:.3f}, '
                f'NSP loss {metric[1] / metric[3]:.3f}')
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break
           
        #print(f'單次epoch的讀取總計: {(now_read - prev_read)/1024}kbs')
        #print(f'單次epoch的寫入總計: {(now_write - prev_write)/1024}kbs')

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')

def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki_2('./wikitext-2/wiki.train.tokens',batch_size, max_len)

mr = MemoryRecord()
mr.AddCheckPoint()
net = BERTModel(len(vocab), num_hiddens=768, norm_shape=[768],
                    ffn_num_input=768, ffn_num_hiddens=256, num_heads=12,
                    num_layers=12, dropout=0.2, key_size=768, query_size=768,
                    value_size=768, hid_in_features=768, mlm_in_features=768,
                    nsp_in_features=768)
mr.AddCheckPoint()
mr.ShowDetail()
devices = try_all_gpus()
loss = nn.CrossEntropyLoss()

train_bert(train_iter, net, loss, len(vocab), devices, 100000)

tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# 词元：'<cls>','a','crane','is','flying','<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]

tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# 词元：'<cls>','a','crane','driver','came','<sep>','he','just',
# 'left','<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
print(encoded_pair_cls)
print(encoded_pair_crane)