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
relation_num = {'P196': 137, 'P1408': 69, 'P195': 136, 'P2633': 176, 'P376': 204, 'P110': 23, 'P115': 29, 'P1731': 114,
                 'P180': 120, 'P523': 250, 'P571': 265, 'P1196': 37, 'P1382': 64, 'P111': 24, 'P206': 147, 'P1327': 54,
                 'P547': 256, 'P737': 305, 'P410': 215, 'P2541': 169, 'P828': 325, 'P669': 287, 'P1478': 84, 'P569': 262,
                 'P205': 146, 'P1552': 93, 'P417': 221, 'P263': 174, 'P131': 49, 'P1040': 9, 'P840': 329, 'P81': 321,
                 'P504': 244, 'P289': 188, 'P1165': 31, 'P291': 189, 'P84': 328, 'P279': 184, 'P2596': 172, 'P706': 297,
                 'P26': 173, 'P400': 210, 'P200': 144, 'P408': 214, 'P629': 280, 'P194': 135, 'P140': 68, 'P1249': 41,
                 'P915': 341, 'P1906': 134, 'P2289': 158, 'P1181': 33, 'P2155': 154, 'P159': 101, '_UNKNOWN': 353,
                 'P1811': 122, 'P2152': 153, 'P674': 288, 'P1192': 36, 'P469': 237, 'P184': 124, 'P40': 209, 'P1302': 46,
                 'P1455': 82, 'P1191': 35, 'P406': 212, 'P241': 164, 'P1001': 2, 'P57': 263, 'P460': 231, 'P427': 224,
                 'P1622': 106, 'P1809': 121, 'P287': 187, 'P1479': 85, 'P1412': 72, 'P748': 307, 'P86': 331, 'P767': 310,
                 'P641': 281, 'P913': 340, 'P1435': 80, 'P201': 145, 'P1308': 48, 'P1365': 62, 'P114': 27, 'P1654': 108,
                 'P1434': 79, 'P541': 255, 'P186': 126, 'P770': 313, 'P485': 239, 'P512': 246, 'P16': 102, 'P97': 349,
                 'P371': 202, 'P126': 42, 'P157': 96, 'P1363': 61, 'P1142': 28, 'P178': 118, 'P577': 267, 'P551': 257,
                 'P85': 330, 'P780': 314, 'P407': 213, 'P658': 284, 'P361': 198, 'P179': 119, 'P1056': 12, 'P425': 223,
                 'P20': 143, 'P121': 38, 'P462': 233, 'P272': 180, 'P609': 272, 'P98': 351, 'P27': 179, 'P58': 268,
                 'P802': 319, 'P30': 190, 'P1411': 71, 'P1038': 8, 'P155': 92, 'P208': 149, 'P87': 333, 'P166': 109,
                 'P1057': 13, 'P1383': 65, 'P190': 133, 'P37': 201, 'P437': 225, 'P788': 315, 'P1962': 138, 'P750': 308,
                 'P2348': 161, 'P38': 205, 'P562': 260, 'P112': 25, 'P452': 229, 'P937': 346, 'P1995': 142, 'P1414': 73,
                 'P1049': 10, 'P655': 283, 'P170': 112, 'P927': 344, 'P1158': 30, 'P6': 270, 'P1322': 53, 'P1875': 127,
                 'P237': 162, 'P197': 139, 'P703': 296, 'P1080': 21, 'P135': 59, 'P790': 316, 'P1064': 15, 'P1589': 100,
                 'P1068': 16, 'P2341': 160, 'P533': 253, 'P620': 279, 'P69': 294, 'P399': 208, 'P567': 261, 'P612': 276,
                 'P54': 254, 'P1313': 51, 'P991': 352, 'P1571': 97, 'P885': 337, 'P1336': 55, 'P825': 323, 'P500': 242,
                 'P880': 336, 'P769': 312, 'P1557': 94, 'P282': 185, 'P575': 266, 'P1081': 22, 'P1018': 4, 'P681': 290,
                 'P277': 183, 'P676': 289, 'P375': 203, 'P1027': 6, 'P2293': 159, 'P88': 335, 'P559': 259, 'P108': 20,
                 'P35': 194, 'P610': 274, 'P1074': 19, 'P931': 345, 'P1429': 76, 'P450': 227, 'P275': 181, 'P1399': 67,
                 'P9': 338, 'P740': 306, 'P17': 111, 'P607': 271, 'P684': 291, 'P1268': 43, 'P421': 222, 'P25': 166,
                 'P2512': 168, 'P127': 45, 'P1574': 98, 'P162': 105, 'P1560': 95, 'P570': 264, 'P22': 156, 'P831': 326,
                 'P1889': 129, 'P449': 226, 'P1416': 74, 'P618': 277, 'P397': 207, 'P412': 217, 'P1389': 66, 'P1072': 18,
                 'P2079': 148, 'P501': 243, 'P688': 292, 'P516': 247, 'P461': 232, 'P1582': 99, 'P0': 1, 'P66': 285,
                 'P1547': 91, 'P793': 317, 'P725': 299, 'P807': 320, 'P366': 200, 'P118': 32, 'P870': 334, 'P509': 245,
                 'P800': 318, 'P161': 103, 'P59': 269, 'P123': 40, 'P276': 182, 'P414': 219, 'P1303': 47, 'P1431': 77,
                 'P735': 304, 'P467': 236, 'P47': 238, 'P941': 347, 'ALL_ZERO': 0, 'P176': 116, 'P730': 302, 'P169': 110,
                 'P306': 191, 'P463': 234, 'P765': 309, 'P413': 218, 'P411': 216, 'P134': 56, 'P360': 197, 'P209': 150,
                 'P2632': 175, 'P172': 113, 'P1317': 52, 'P144': 81, 'P1851': 125, 'P664': 286, 'P113': 26, 'P19': 132,
                 'P1532': 89, 'P611': 275, 'P1312': 50, 'P1344': 57, 'P826': 324, 'P734': 303, 'P7': 295, 'P1346': 58,
                 'P163': 107, 'P2184': 155, 'P457': 230, 'P2416': 165, 'P1891': 131, 'P729': 301, 'P1619': 104, 'P2670': 178,
                 'P517': 248, 'P122': 39, 'P355': 195, 'P726': 300, 'P2389': 163, 'P141': 70, 'P138': 63, 'P39': 206,
                 'P530': 252, 'P189': 130, 'P1885': 128, 'P92': 342, 'P451': 228, 'P1990': 141, 'P1071': 17, 'P1420': 75,
                 'P647': 282, 'P868': 332, 'P136': 60, 'P344': 193, 'P466': 235, 'P972': 350, 'P619': 278, 'P286': 186,
                 'P50': 241, 'P149': 86, 'P708': 298, 'P403': 211, 'P53': 251, 'P2505': 167, 'P36': 196, 'P119': 34,
                 'P177': 117, 'P264': 177, 'P103': 7, 'P2286': 157, 'P102': 5, 'P175': 115, 'P61': 273, 'P150': 87,
                 'P31': 192, 'P2546': 170, 'P21': 152, 'P106': 14, 'P495': 240, 'P921': 343, 'P2098': 151, 'P1531': 88,
                 'P364': 199, 'P689': 293, 'P415': 220, 'P101': 3, 'P1535': 90, 'P199': 140, 'P1433': 78, 'P832': 327,
                 'P2578': 171, 'P822': 322, 'P91': 339, 'P1462': 83, 'P522': 249, 'P183': 123, 'P945': 348, 'P1050': 11,
                 'P553': 258, 'P1269': 44, 'P768': 311}

all_key = relation_num.keys()
propertylist = []
for i in all_key:
    propertylist.append(i)

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
with open(os.path.join(CORPUS_FOlDER, 'all_selected_data_with_one_edge_per_sentence.json'), 'w') as f:
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
        #new_re_list[str(i + 1)] = re_list[i + 1]
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
with open(os.path.join(CORPUS_FOlDER, 'new_long_data.json'), 'w') as f:
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
    train_num0 = round(temp_len * 0.5)
    train_index0 = value1[0: train_num0]
    for index0 in train_index0:
        new_train.append(new_file[index0])
    eval_num0 = round(temp_len * 0.25)
    eval_index0 = value1[train_num0: train_num0 + eval_num0]
    for index0 in eval_index0:
        new_eval.append(new_file[index0])
    #test_num0 = round(temp_len * 0.5)
    test_index0 = value1[train_num0 + eval_num0:]
    for index0 in test_index0:
        new_test.append(new_file[index0])

random.shuffle(new_train)
random.shuffle(new_test)
random.shuffle(new_eval)
with open('./all_preprocessed_data/train_data_single_sentences.json', 'w') as f:
    json.dump(new_train, f)
with open('./all_preprocessed_data/test_data_single_sentences.json', 'w') as f:
    json.dump(new_test, f)
with open('./all_preprocessed_data/eval_data_single_sentences.json', 'w') as f:
    json.dump(new_eval, f)