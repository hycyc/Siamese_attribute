import os
import numpy as np
import tensorflow as tf
import random
import sys
import json
import _pickle as cPickle
from core import embeddings


embedding_matrix, word2idx = embeddings.load('./glove/glove.6B.50d.txt')
print("Loaded embeddings:", embedding_matrix.shape)
### preprocess all the wiki data and select desired data to construct data pairs
CORPUS_FOlDER = "./enwiki-20160501"
propertylist = ['P20', 'P19', 'P551', 'P463', 'P108', 'P157', 'P69', 'P172', 'P140',
                'P26', 'P40', 'P22', 'P25', 'P119', 'P66', 'P27', 'P101', 'P800', 'P166',
                'P39', 'P102', 'P263', 'P184', 'P802', 'P53', 'P553', 'P1344', 'P1416', 'P103',
                'P91', 'P237', 'P411', 'P412', 'P450', 'P97', 'P512', 'P1303', 'P1399', 'P1412',
                'P1429', 'P451', 'P1038', 'P21', 'P734', 'P735', 'P570', 'P569', 'P1196', 'P106', 'P509']

relation_num = {'P20': 50, 'P19': 1, 'P551': 2, 'P463': 3, 'P108': 4, 'P157': 5, 'P69': 6,
                'P172': 7, 'P140': 8, 'P26': 9, 'P40': 10, 'P22': 11, 'P25': 12, 'P119': 13,
                'P66': 14, 'P27': 15, 'P101': 16, 'P800': 17, 'P166': 18, 'P39': 19, 'P102': 20,
                'P263': 21, 'P184': 22, 'P802': 23, 'P53': 24, 'P553': 25, 'P1344': 26,
                'P1416': 27, 'P103': 28, 'P91': 29, 'P237': 30, 'P411': 31, 'P412': 32, 'P450': 33,
                'P97': 34, 'P512': 35, 'P1303': 36, 'P1399': 37, 'P1412': 38, 'P1429': 39, 'P451': 40,
                'P1038': 41, 'P21': 42, 'P734': 43, 'P735': 44, 'P570': 45, 'P569': 46,
                'P1196': 47, 'P106': 48, 'P509': 49, 'ALL_ZERO': 0, '_UNKNOWN': 51}

with open(os.path.join(CORPUS_FOlDER, 'semantic-graphs-filtered-training.02_06.json')) as f:
    file_data1 = json.load(f)
with open(os.path.join(CORPUS_FOlDER, 'semantic-graphs-filtered-validation.02_06.json')) as f:
    file_data2 = json.load(f)
with open(os.path.join(CORPUS_FOlDER, 'semantic-graphs-filtered-held-out.02_06.json')) as f:
    file_data3 = json.load(f)
file_data = file_data1 + file_data2 + file_data3
print('all data loaded')
# 筛选出原语料库中只含有propertylist的句子
sentence_num = len(file_data)
new_file = []
ids_in_sentence = []
for s, sentence in enumerate(file_data):
    if 'edgeSet' in sentence:
        edge_data = sentence['edgeSet']
        temp_edges = []
        for _, edge in enumerate(edge_data):
            temp_kb_id = edge['kbID']
            if temp_kb_id in propertylist:
                temp_sentence1 = {'edgeSet': [], 'tokens': []}
                temp_sentence1['edgeSet'].append(edge)
                temp_sentence1['tokens'] = sentence['tokens']
                temp_sentence1['id'] = s
                new_file.append(temp_sentence1)
# 原语料经过筛选后只含有propertylist的句子的结果
### 得到所有的句子，并每个句子只包含一种关系
with open(os.path.join(CORPUS_FOlDER, 'selected_data_with_one_edge_per_sentence.json'), 'w') as f:
    json.dump(new_file, f)
print('selected sentence length is:')
print(len(new_file))
### relation-sentence_list（re_list）是一个键为关系编号，值为其存在的句子编号列表的字典(list1)
relation_ordered = sorted(relation_num.items(), key=lambda item: item[1])  # 将属性表排成正序。注：返回的是元组元素的列表！
re_list = dict.fromkeys(range(1, len(relation_ordered)-1), [])  # 将属性和句子的对应先建立一个value默认的字典（之后直接给相应的键赋值即可）'
len_record = []
for i in range(1, len(relation_ordered)-1):
    temp_num = []
    for num, sentence_data in enumerate(new_file):
        if 'edgeSet' in sentence_data:
            edge_context = sentence_data['edgeSet']
            kb_id = edge_context[0]['kbID']
            if kb_id == relation_ordered[i][0]:
                temp_num.append(num)
    re_list[i] = temp_num
    len_record.append(len(temp_num))

## 去除掉那些句子数量少于100的关系
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]
short_record = []
long_record = []
long_length = []
new_re_list = {}
short_length = []
for i, j in enumerate(len_record):
    temp_key = get_key(relation_num, i + 1)
    if j < 100:
        short_record.append(temp_key)
        short_length.append(j)
    else:
        long_record.append(temp_key)
        long_length.append(j)
        new_re_list[str(i+1)] = re_list[i+1]
print('relations with more sentences:')
print(long_record)
print('number of sentences with more sentences:')
print(long_length)
print(len(new_re_list))
print(short_length)
print(short_record)
with open(os.path.join(CORPUS_FOlDER, 'long_data1.json'), 'w') as f:
    json.dump(new_re_list, f)
# 获取所有句子数量大于100的关系的kbid
valid_kbid = long_record
# new_file1 = []
# long_record1 = []
# for lr1 in long_record:
#     long_record1.append(lr1[0])
# # only sentences of relation with more sentences are left
# for sent in new_file:
#     if sent['edgeSet'][0]['kbID'] in long_record1:
#         new_file1.append(sent)
# print(len(new_file1))
# new_file = new_file1
# with open('./preprocessed_data211/train_data_single_sentences.json') as f:
#     train1 = json.load(f)
# with open('./preprocessed_data211/test_data_single_sentences.json') as f:
#     test1 = json.load(f)
# with open('./preprocessed_data211/eval_data_single_sentences.json') as f:
#     eval1 = json.load(f)
# separate into train test and eval
#new_file = train1 + test1 + eval1
new_train = []
new_test = []
new_eval = []
items01 = new_re_list.items()
for key1, value1 in items01:
    temp_len = len(value1)
    random.shuffle(value1)
    train_num0 = round(temp_len * 0.8)
    train_index0 = value1[0: train_num0]
    for index0 in train_index0:
        new_train.append(new_file[index0])
    eval_num0 = round(temp_len * 0.1)
    eval_index0 = value1[train_num0: train_num0 + eval_num0]
    for index0 in eval_index0:
        new_eval.append(new_file[index0])
    #test_num0 = round(temp_len * 0.5)
    test_index0 = value1[train_num0 + eval_num0:]
    for index0 in test_index0:
        new_test.append(new_file[index0])

with open('./preprocessed_data811/train_data_single_sentences.json', 'w') as f:
    json.dump(new_train, f)
with open('./preprocessed_data811/test_data_single_sentences.json', 'w') as f:
    json.dump(new_test, f)
with open('./preprocessed_data811/eval_data_single_sentences.json', 'w') as f:
    json.dump(new_eval, f)
# with open('../data/train_data_single_sentences.json') as f:
#     new_train = json.load(f)
# with open('../data/test_data_single_sentences.json') as f:
#     new_test = json.load(f)
# with open('../data/eval_data_single_sentences.json') as f:
#     new_eval = json.load(f)
for s in range(len(new_train)):
    print(s)
    temp_sentence = new_train[s]
    temp_sentence['same_kbid'] = []
    temp_sentence['diff_kbid'] = []
    for s1 in range(len(new_train)):
        temp_sentence1 = new_train[s1]
        if temp_sentence['id'] == temp_sentence1['id'] and s1 != s: #是同一个句子
            if temp_sentence['edgeSet'][0]['kbID'] == temp_sentence1['edgeSet'][0]['kbID']:#是同一种关系
                temp_sentence['same_kbid'].append(s1)
            else:#不同关系
                temp_sentence['diff_kbid'].append(s1)
    new_train[s] = temp_sentence
for s in range(len(new_test)):
    print(s)
    temp_sentence = new_test[s]
    temp_sentence['same_kbid'] = []
    temp_sentence['diff_kbid'] = []
    for s1 in range(len(new_test)):
        temp_sentence1 = new_test[s1]
        if temp_sentence['id'] == temp_sentence1['id'] and s1 != s: #是同一个句子
            if temp_sentence['edgeSet'][0]['kbID'] == temp_sentence1['edgeSet'][0]['kbID']:#是同一种关系
                temp_sentence['same_kbid'].append(s1)
            else:#不同关系
                temp_sentence['diff_kbid'].append(s1)
    new_test[s] = temp_sentence
for s in range(len(new_eval)):
    print(s)
    temp_sentence = new_eval[s]
    temp_sentence['same_kbid'] = []
    temp_sentence['diff_kbid'] = []
    for s1 in range(len(new_eval)):
        temp_sentence1 = new_eval[s1]
        if temp_sentence['id'] == temp_sentence1['id'] and s1 != s: #是同一个句子
            if temp_sentence['edgeSet'][0]['kbID'] == temp_sentence1['edgeSet'][0]['kbID']:#是同一种关系
                temp_sentence['same_kbid'].append(s1)
            else:#不同关系
                temp_sentence['diff_kbid'].append(s1)
    new_eval[s] = temp_sentence
# get the new relist of each data for train test and eval
selected_relations = {}
for i in range(len(long_record)):
    selected_relations[long_record[i][0]] = i
relation_ordered0 = sorted(selected_relations.items(), key=lambda item: item[1])  # 将属性表排成正序。注：返回的是元组元素的列表！
train_re_list = dict.fromkeys(range(len(relation_ordered0)), [])  # 将属性和句子的对应先建立一个value默认的字典（之后直接给相应的键赋值即可）'
test_re_list = dict.fromkeys(range(len(relation_ordered0)), [])
eval_re_list = dict.fromkeys(range(len(relation_ordered0)), [])
train_len_record = []
test_len_record = []
eval_len_record = []
for i in range(len(relation_ordered0)):
    train_num = []
    test_num = []
    eval_num = []
    for num, sentence_data in enumerate(new_train):
        if 'edgeSet' in sentence_data:
            edge_context = sentence_data['edgeSet']
            kb_id = edge_context[0]['kbID']
            if kb_id == relation_ordered0[i][0]:
                train_num.append(num)
    train_re_list[i] = train_num
    train_len_record.append(len(train_num))
    for num, sentence_data in enumerate(new_test):
        if 'edgeSet' in sentence_data:
            edge_context = sentence_data['edgeSet']
            kb_id = edge_context[0]['kbID']
            if kb_id == relation_ordered0[i][0]:
                test_num.append(num)
    test_re_list[i] = test_num
    test_len_record.append(len(test_num))
    for num, sentence_data in enumerate(new_eval):
        if 'edgeSet' in sentence_data:
            edge_context = sentence_data['edgeSet']
            kb_id = edge_context[0]['kbID']
            if kb_id == relation_ordered0[i][0]:
                eval_num.append(num)
    eval_re_list[i] = eval_num
    eval_len_record.append(len(eval_num))
##对于每一种关系的每一个句子，判断是否有句子与之共享原始句子，包括同中关系的句子和异种关系的句子
#####接下来要做的是，1.组成pair；2.变换为embedding；3.写入tfrecorder。
#select the relations with more sentences and construct positive and negative data in a balanced mode
all_tuple_list = []
#posi_neg_num = len(long_record)
posi_neg_num = 40
items1 = train_re_list.items()
print('begin train data')
for re_id, relation in items1:
    re_id00 = str(re_id)
    #relation is a list
    for s, s_id in enumerate(relation):
        temp_sentence = new_train[s_id]
        same_id_len = len(temp_sentence['same_kbid'])
        if same_id_len > 0:
            if same_id_len > posi_neg_num:
                new_relation = random.sample(temp_sentence['same_kbid'], posi_neg_num)
                for _, s1_id in enumerate(new_relation):
                    all_tuple_list.append([s_id, s1_id, 1])
            else:
                num_diff = posi_neg_num - same_id_len
                new_relation = temp_sentence['same_kbid']
                new_relation0 = random.sample(relation, num_diff)
                new_relation = new_relation + new_relation0
                for _, s1_id in enumerate(new_relation):
                    all_tuple_list.append([s_id, s1_id, 1])
        else:
            new_relation = random.sample(relation, posi_neg_num)
            for _, s1_id in enumerate(new_relation):
                all_tuple_list.append([s_id, s1_id, 1])
        all_keys = list(train_re_list.keys())
        other_keys = [i for i in all_keys if i != re_id]
        new_dict = {}
        for key in other_keys:
            new_dict[key] = train_re_list[key]
        items0 = new_dict.items()
        #items2 = random.sample(items0, posi_neg_num)
        diff_id_len = len(temp_sentence['diff_kbid'])
        if diff_id_len > 0:
            if diff_id_len > posi_neg_num:
                new_relation1 = random.sample(temp_sentence['diff_kbid'], posi_neg_num)
                for _, s1_id in enumerate(new_relation1):
                    all_tuple_list.append([s_id, s1_id, 0])
            else:
                num_diff = posi_neg_num - diff_id_len
                new_relation = temp_sentence['diff_kbid']
                items2 = random.sample(items0, num_diff)
                for _, relation1 in items2:
                    new_relation = new_relation + random.sample(relation1, 1)
                for _, s1_id in enumerate(new_relation):
                    all_tuple_list.append([s_id, s1_id, 0])
        else:
            items2 = random.sample(items0, posi_neg_num)
            for _, relation1 in items2:
                new_relation1 = random.sample(relation1, 1)
                all_tuple_list.append([s_id, new_relation1[0], 0])
print('all train pairs found, begin writing')
random.shuffle(all_tuple_list)
print('train len is:', len(all_tuple_list))
with open('./our_data811/train_pairs.json', 'w') as f:
    json.dump(all_tuple_list, f)
all_tuple_list = []
#posi_neg_num = len(long_record)
posi_neg_num = 10
items1 = eval_re_list.items()
print('begin eval data')
for re_id, relation in items1:
    re_id00 = str(re_id)
    #relation is a list
    for s, s_id in enumerate(relation):
        temp_sentence = new_eval[s_id]
        same_id_len = len(temp_sentence['same_kbid'])
        if same_id_len > 0:
            if same_id_len > posi_neg_num:
                new_relation = random.sample(temp_sentence['same_kbid'], posi_neg_num)
                for _, s1_id in enumerate(new_relation):
                    all_tuple_list.append([s_id, s1_id, 1])
            else:
                num_diff = posi_neg_num - same_id_len
                new_relation = temp_sentence['same_kbid']
                new_relation0 = random.sample(relation, num_diff)
                new_relation = new_relation + new_relation0
                for _, s1_id in enumerate(new_relation):
                    all_tuple_list.append([s_id, s1_id, 1])
        else:
            new_relation = random.sample(relation, posi_neg_num)
            for _, s1_id in enumerate(new_relation):
                all_tuple_list.append([s_id, s1_id, 1])
        all_keys = list(eval_re_list.keys())
        other_keys = [i for i in all_keys if i != re_id]
        new_dict = {}
        for key in other_keys:
            new_dict[key] = eval_re_list[key]
        items0 = new_dict.items()
        #items2 = random.sample(items0, posi_neg_num)
        diff_id_len = len(temp_sentence['diff_kbid'])
        if diff_id_len > 0:
            if diff_id_len > posi_neg_num:
                new_relation1 = random.sample(temp_sentence['diff_kbid'], posi_neg_num)
                for _, s1_id in enumerate(new_relation1):
                    all_tuple_list.append([s_id, s1_id, 0])
            else:
                num_diff = posi_neg_num - diff_id_len
                new_relation = temp_sentence['diff_kbid']
                items2 = random.sample(items0, num_diff)
                for _, relation1 in items2:
                    new_relation = new_relation + random.sample(relation1, 1)
                for _, s1_id in enumerate(new_relation):
                    all_tuple_list.append([s_id, s1_id, 0])
        else:
            items2 = random.sample(items0, posi_neg_num)
            for _, relation1 in items2:
                new_relation1 = random.sample(relation1, 1)
                all_tuple_list.append([s_id, new_relation1[0], 0])
print('all eval pairs found, begin writing')
random.shuffle(all_tuple_list)
with open('./our_data811/eval_pairs.json', 'w') as f:
    json.dump(all_tuple_list, f)
print('eval len is:', len(all_tuple_list))
