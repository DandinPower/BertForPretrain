import os
import random
import torch
from d2l import torch as d2l

#讀取dataset
def _read_wiki(data_dir):
    file_name = data_dir
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 大寫轉小寫並篩選出有兩句以上的文章
    paragraphs = [line.strip().lower().split(' . ')for line in lines if len(line.split('.')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

#產生next_sentence任務
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考慮特殊字元<cls>,2*<sep>
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # 創建新的mask副本,可能替換成<mask>或隨機字元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 使得mlm的隨機出現位置被打亂
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%被替換成masj
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10趴的時間不變
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的時間用隨機字元取代
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一個字符串列表
    for i, token in enumerate(tokens):
        # 在masklm不會去處理特殊字元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 根據文獻以15%遮蔽文本
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        #valid_lens不包括<pad>
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        #<pad>的部分將會在權重 = 0被濾掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 取得nsp任務
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 取得masklm任務
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # padding
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

#@save
def load_data_wiki_2(datapath,batch_size, max_len):
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(datapath)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab

def main():
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki_2('./data/wikitext-2/wiki.train.tokens',batch_size, max_len)
    print(train_iter)
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
        mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
            pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
            nsp_y.shape)
        print('token_x')
        print('/////////////////////')
        print(tokens_X)
        print('segments_x')
        print('/////////////////////')
        print(segments_X)
        print('valid_lens_x')
        print('/////////////////////')
        print(valid_lens_X)
        print('pred_positions_x')
        print('/////////////////////')
        print(pred_positions_X)
        print('mlm_weights_x')
        print('/////////////////////')
        print(mlm_weights_X)
        print('mlm_Y')
        print('/////////////////////')
        print(mlm_Y)
        print('nsp_Y')
        print('/////////////////////')
        print(nsp_Y)
        
        break
    print(len(vocab))
    '''
    train_iter, vocab = load_data_wiki_2('./data/wikidata/data.txt',batch_size, max_len)
    print(train_iter)
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
        mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
            pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
            nsp_y.shape)
        break
    print(len(vocab))'''

if __name__ == '__main__':
    main()