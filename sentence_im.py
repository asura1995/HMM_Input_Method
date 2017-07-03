#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import os
import re
import struct
from collections import defaultdict
from functools import reduce
from math import log


def read_py_word_dict():
    py_word_dict = defaultdict(list)
    with codecs.open('LM.Dic', 'r', 'gbk') as fr:
        lm_dic_content = fr.read()
    word_list = re.findall(r'#(.*)\n', lm_dic_content)
    temp_py_list = re.findall(r'PY:(.*)\n', lm_dic_content)
    py_list = [[y[:-1] for y in x.strip().split(' ')] for x in temp_py_list]
    for i in range(len(py_list)):
        pinyin = '_'.join(py_list[i])
        word = word_list[i]
        py_word_dict[pinyin].append(word)
    return word_list, py_word_dict


def cut_files():
    train_files_list = []
    line_num = 0
    tmp_list = []
    file_name_num = 10
    if not os.path.exists('train_files'):
        os.makedirs('train_files')
    with codecs.open('train', 'r', 'gbk') as fr:
        for line in fr:
            if line_num >= 1000:
                file_name = 'train_files/' + str(file_name_num) + '.txt'
                train_files_list.append(file_name)
                with codecs.open(file_name, 'w', 'utf-8') as fw:
                    for t in tmp_list:
                        fw.write(t + '\n')
                file_name_num += 1
                line_num = 0
                tmp_list = []
            tmp_list.append(line.strip())
    if line_num < 1000:
        file_name = 'train_files/' + str(file_name_num) + '.txt'
        train_files_list.append(file_name)
        with codecs.open(file_name, 'w', 'utf-8') as fw1:
            for t in tmp_list:
                fw1.write(t + '\n')
    return train_files_list


def segment(word_set, file_a):
    with codecs.open(file_a, 'r', 'utf-8') as fr:
        sentences = fr.read().split('\n')
    lines = []
    for s in sentences:
        tmp_line = re.findall(r'\w+', s)
        lines.extend(tmp_line)
    word_list = []
    for line in lines:
        word_in_line = []
        while len(line) > 0:
            a_border = 0
            b_border = len(line)
            for i in range(len(line[a_border:b_border])):
                tmp_word = line[a_border + i: b_border]
                if tmp_word in word_set or len(tmp_word) == 1:
                    word_in_line.insert(0, tmp_word)
                    split_num = a_border + i
                    break
            line = line[:split_num]
        word_in_line.insert(0, '^BEG')
        word_in_line.append('$END')
        word_list.extend(word_in_line)
    return word_list


def train_unigram(word_set, file_a):
    tmp_unigram_dict = {}
    segment_words = segment(word_set, file_a)
    for word in segment_words:
        if word not in tmp_unigram_dict:
            tmp_unigram_dict[word] = 0
        tmp_unigram_dict[word] += 1
    return tmp_unigram_dict


def train_bigram(word_set, file_a):
    tmp_bigram_dict = {}
    segment_words = segment(word_set, file_a)
    for i in range(1, len(segment_words)):
        bigram = segment_words[i - 1] + '_' + segment_words[i]
        if bigram == "$END_^BEG":
            continue
        if bigram not in tmp_bigram_dict:
            tmp_bigram_dict[bigram] = 0
        tmp_bigram_dict[bigram] += 1
    return tmp_bigram_dict


def join_dicts(dict_a, dict_b):
    for k, v in dict_b.items():
        if k in dict_a.keys():
            dict_a[k] += v
        else:
            dict_a[k] = v
    return dict_a


def calc_bi_prob(unigram_dict, bigram_dict):
    log_bigram_dict = {}
    length = len(unigram_dict)
    for k, v in bigram_dict.items():
        a = k.split('_')[0]
        if a == '':
            a = '_'
        log_bigram_dict[k] = log(
            (v + 0.01) / (unigram_dict[a] + 0.01 * length))
    log_bigram_dict['^ZERO$'] = log(0.01 / (0.01 * length))
    return log_bigram_dict


def write_grams(dict_a, filename):
    try:
        with open(filename, 'wb') as fw:
            fw.write(struct.pack('<L', len(dict_a)))
            for word, freq in dict_a.items():
                bin_word = word.encode('gbk')
                fw.write(struct.pack('<L', len(bin_word)) + bin_word)
                fw.write(struct.pack('<f', freq))
        return 0
    except:
        return 1


def write_py_word(dict_a, filename):
    try:
        with open(filename, 'wb') as fw:
            fw.write(struct.pack('<L', len(dict_a)))
            for py, words in dict_a.items():
                bin_py = py.encode('gbk')
                fw.write(struct.pack('<L', len(bin_py)) + bin_py)
                fw.write(struct.pack('<L', len(words)))
                for word in words:
                    bin_word = word.encode('gbk')
                    len_word = struct.pack('<L', len(bin_word))
                    fw.write(len_word + bin_word)
        return 0
    except:
        return 1


def train():
    print('Reading From LM.Dic...')
    word_list, py_word_dict = read_py_word_dict()
    print('Cutting train files...')
    train_files_list = cut_files()
    word_set = set(word_list)
    print('Mapping...')
    train_unigram_dict_list = map(
        lambda x: train_unigram(word_set, x), train_files_list)
    train_bigram_dict_list = map(
        lambda x: train_bigram(word_set, x), train_files_list)
    print('Reducing...')
    unigram_dict = reduce(join_dicts, train_unigram_dict_list)
    bigram_dict = reduce(join_dicts, train_bigram_dict_list)
    print('Calculating Probabilities...')
    log_bigram_dict = calc_bi_prob(unigram_dict, bigram_dict)
    print('Write to file...')
    write_grams(log_bigram_dict, "bigram_dict.dat")
    write_py_word(py_word_dict, "py_word_dict.dat")
    print('Done')
    return log_bigram_dict, py_word_dict


def read_grams_from_dict(file_name):
    with open(file_name, 'rb') as fr:
        file_data = fr.read()
    dict_from_file = {}
    offset = 0
    bin_len_of_dict = file_data[offset:offset + 4]
    offset += 4
    len_of_dict = struct.unpack('<L', bin_len_of_dict)[0]
    print('Length of Dict: ', len_of_dict)
    for i in range(len_of_dict):
        bin_len_of_word = file_data[offset:offset + 4]
        offset += 4
        len_of_word = struct.unpack('<L', bin_len_of_word)[0]
        bin_word = file_data[offset:offset + len_of_word]
        offset += len_of_word
        word = bin_word.decode('gbk')
        bin_freq = file_data[offset:offset + 4]
        offset += 4
        freq = struct.unpack('<f', bin_freq)[0]
        dict_from_file[word] = freq
    return dict_from_file


def read_py_word_dict_from_file(file_name):
    with open(file_name, 'rb') as fr:
        file_data = fr.read()
    dict_from_file = {}
    offset = 0
    bin_len_of_dict = file_data[offset:offset + 4]
    offset += 4
    len_of_dict = struct.unpack('<L', bin_len_of_dict)[0]
    print('Length of Dict: ', len_of_dict)
    for i in range(len_of_dict):
        bin_len_of_py = file_data[offset:offset + 4]
        offset += 4
        len_of_py = struct.unpack('<L', bin_len_of_py)[0]
        bin_py = file_data[offset:offset + len_of_py]
        offset += len_of_py
        py = bin_py.decode('gbk')
        bin_len_of_words = file_data[offset:offset + 4]
        offset += 4
        len_of_words = struct.unpack('<L', bin_len_of_words)[0]
        words = []
        for i in range(len_of_words):
            bin_len_of_word = file_data[offset:offset + 4]
            offset += 4
            len_of_word = struct.unpack('<L', bin_len_of_word)[0]
            bin_word = file_data[offset:offset + len_of_word]
            offset += len_of_word
            word = bin_word.decode('gbk')
            words.append(word)
        dict_from_file[py] = words
    return dict_from_file


def viterbi(pinyin, py_word_dict, bigram_freq_dict):
    candidates = []
    magnet_pinyin = []
    # 建立磁贴
    for i in range(len(pinyin)):
        tmp_py = pinyin[0:i + 1]
        tmp_zi_dict = {}
        while len(tmp_py) >= 1:
            test_py = '_'.join(tmp_py)
            if test_py in py_word_dict.keys():
                zi_list = py_word_dict[test_py]
                for zi in zi_list:
                    tmp_zi_dict[zi] = [0, '']
            tmp_py = tmp_py[1:]
        magnet_pinyin.append(tmp_zi_dict)

    magnet_pinyin.insert(0, {'^BEG': [0, '']})
    magnet_pinyin.append({'$END': [0, '']})

    # 计算磁贴中的值
    for i in range(1, len(magnet_pinyin)):
        for cur_word in magnet_pinyin[i].keys():
            if cur_word == '$END':
                pre_index = i - 1
            else:
                pre_index = i - len(cur_word)
            cur_value = -10000
            cur_point = ''
            for pre_word in magnet_pinyin[pre_index].keys():
                pre_value = magnet_pinyin[pre_index][pre_word][0]
                bigram = pre_word + '_' + cur_word
                if bigram in bigram_freq_dict.keys():
                    tmp_value = bigram_freq_dict[bigram] + pre_value
                else:
                    tmp_value = bigram_freq_dict['^ZERO$'] + pre_value
                if tmp_value > cur_value:
                    cur_value = tmp_value
                    cur_point = pre_word
            magnet_pinyin[i][cur_word][0] = cur_value
            magnet_pinyin[i][cur_word][1] = cur_point

    # 回退
    last_words_list = sorted(
        magnet_pinyin[-2].keys(), key=lambda x: magnet_pinyin[-2][x][0],
        reverse=True)
    for word in last_words_list:
        i = -2
        prev = ''
        tmp_str = word
        key_word = word
        while prev != '^BEG':
            prev = magnet_pinyin[i].get(key_word)[1]
            tmp_str = prev + tmp_str
            i = i - len(key_word)
            key_word = prev
        tmp_str = tmp_str[4:]
        if tmp_str not in candidates:
            candidates.append(tmp_str)
    return candidates


def show_im(py_word_dict, bigram_freq_dict, cls_info):
    hanzi_str = ''
    pinyin_str = ''
    while 1:
        os.system(cls_info)
        print(hanzi_str)
        pinyin_str = input("(quit)|(del)={Enter}|reinput(#)-->")
        pinyin = pinyin_str.split(' ')
        if pinyin_str == '' or pinyin_str == 'del':
            hanzi_str = hanzi_str[:-1]
            continue
        elif pinyin_str[-1] == '#':
            continue
        elif len(pinyin_str) == 1:
            if ord(pinyin_str) == 32:
                hanzi_str += chr(12288)
                continue
            elif ord(pinyin_str) == 46:
                hanzi_str += chr(12290)
                continue
            elif ord(pinyin_str) >= 32 and ord(pinyin_str) <= 126:
                hanzi_str += chr(ord(pinyin_str) + 65248)
                continue
            else:
                continue
        elif pinyin_str == 'quit':
            break

        candidates = viterbi(pinyin, py_word_dict, bigram_freq_dict)
        n = 0
        while 1:
            tmp_list = []
            i = 0
            for w in candidates[n:n + 5]:
                print('【%d】%s' % (i + 1, w), end='')
                tmp_list.append(w)
                i += 1
            print('\n')
            candi_num = input('|<--(p) (r) (n)-->|Num:')
            if candi_num == 'p':
                n -= 5
                continue
            elif candi_num == 'n':
                n += 5
                continue
            elif candi_num == 'r':
                break
            elif candi_num == '':
                hanzi_str += tmp_list[0] if tmp_list != [] else ''
                break
            elif candi_num >= '1' and candi_num <= '5':
                hanzi_str += tmp_list[int(candi_num) - 1]
                break
            else:
                print('Plz input 1-5 or n or p.')
                continue


def core():
    print("Try Loading...")
    try:
        bigram_freq_dict = read_grams_from_dict("bigram_dict.dat")
        print('Bigram Loaded.')
        py_word_dict = read_py_word_dict_from_file("py_word_dict.dat")
        print('Pinyin-Word Loaded.')
        print('Done.')
    except:
        print("Loading Faild. Processing...")
        bigram_freq_dict, py_word_dict = train()
    if os.system('clear'):
        cls_info = 'cls'
    else:
        cls_info = 'clear'
    show_im(py_word_dict, bigram_freq_dict, cls_info)


if __name__ == '__main__':
    core()
