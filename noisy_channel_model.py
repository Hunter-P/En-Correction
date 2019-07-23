# coding:utf-8
# Author:pengjiaxin


import re
from collections import Counter
from sklearn.externals import joblib
import random


def build_words():
    text = open('big.txt').read()
    # 统计词频
    WORDS = Counter(re.findall(r'\w+', text.lower()))
    joblib.dump(WORDS, 'data/WORDS')
    print("WORDS:", WORDS)


def create_wrong_word(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    r1 = random.randint(0, 2)
    if r1 == 0:
        # 随机替换某个字母
        r2 = random.randint(0, len(word)-1)
        r3 = random.randint(0, len(letters)-1)
        word_l = list(word)
        word_l[r2] = letters[r3]
        return ''.join(word_l)
    if r1 == 1:
        # 随机增加某个字母
        r2 = random.randint(0, len(word) - 1)
        r3 = random.randint(0, len(letters) - 1)
        return word[:r2]+letters[r3]+word[r2:]
    if r1 == 2:
        # 随机减去某个字母
        r2 = random.randint(0, len(word) - 1)
        word_l = list(word)
        del word_l[r2]
        return ''.join(word_l)


def add_weight():
    """
    在词典里添加单词
    """
    words_dict = joblib.load('data/WORDS')
    text = open('lc_word.txt').read()
    text = text.strip().split()
    for w in text:
        try:
            if words_dict[w.lower()] < 10000:
                words_dict[w.lower()] = words_dict[w.lower()] + 10000
            else:
                words_dict[w.lower()] = words_dict[w.lower()]
        except KeyError:
            words_dict[w.lower()] = 15000
    words_dict = dict(sorted(words_dict.items(), key=lambda x: x[1], reverse=True))
    del words_dict['cody']
    del words_dict['crf']
    del words_dict['cf']
    del words_dict['chf']
    joblib.dump(words_dict, 'data/words_dict')
    return words_dict


def create_word_dict():
    words_dict = dict()
    text = open('lc_word.txt').read()
    text = text.strip().split()
    for w in text:
        words_dict[w.lower()] = 10
    words_dict = dict(sorted(words_dict.items(), key=lambda x: x[1], reverse=True))
    joblib.dump(words_dict, 'data/new_words_dict')
    return words_dict


class Correction:
    def __init__(self, words_dict):
        self.WORDS = words_dict

    def edits1(self, word):
        """
        编辑距离为1
        :param word:
        :return:
        """
        letters = 'abcdefghijklmnopqrstuvwxyz()'
        splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
        deletes = [l+r[1:] for l, r in splits if r]
        transposes = [l+r[1]+r[0]+r[2:] for l, r in splits if len(r)>1]
        replaces = [l+c+r[1:] for l, r in splits if r for c in letters]
        inserts = [l+c+r for l, r in splits for c in letters]
        return set(deletes+transposes+replaces+inserts)

    def edits2(self, word):
        """
        编辑距离为2
        :param word:
        :return:
        """
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def edits3(self, word):
        """
        编辑距离为3
        :param word:
        :return:
        """
        return (e3 for e1 in self.edits1(word) for e2 in self.edits1(e1) for e3 in self.edits1(e2))

    def known(self, words):
        """
        过滤掉不存在的单词
        :param words:
        :return:
        """
        return set(w for w in words if w in self.WORDS)

    def prob(self, word):
        """
        计算单词的出现概率
        :param word:
        :param N:
        :return:
        """
        N = sum(self.WORDS.values())
        try:
            return self.WORDS[word]/N
        except KeyError:
            return 1

    def correction_word(self, word):
        """
        修正单词，返回概率最大的候选词
        :param word:
        :return:
        """
        lower_word = word.lower()
        if len(word) < 3:  # 如果字符长度小于3，则不进行纠正
            return word
        if word == '(5)':
            return '(s)'
        if len(word) > 20:
            return word
        if not re.findall('[a-zA-Z]+', word):  # 如果字符不包含英文字母，则不进行纠正
            return word
        if '_' in word:
            return ' '.join([self.correction_word(w) for w in word.split('_')])
        if '(' in word in word:
            return '('.join([self.correction_word(w) for w in word.split('(')])
        if word[-1] in ",.;:!)”":
            return self.correction_word(word[:-1]) + word[-1]
        if word[0] in "“":
            return word[0] + self.correction_word(word[1:])
        if '/' in word:
            return '/'.join([self.correction_word(w) for w in word.split("/")])
        if "'" in word:
            return "'".join([self.correction_word(w) for w in word.split("'")])

        c_word = max(self.candidates(lower_word), key=self.prob)
        if word.isupper():
            return c_word.upper()
        if word.istitle():
            return c_word[0].upper() + c_word[1:]
        return c_word

    def correction_sentence(self, sentence):
        """
        修正句子
        :param sentence:
        :return:
        """
        correct_sentence = [self.correction_word(word) for word in sentence]
        return correct_sentence

    def candidates(self, word):
        """
        生成单词候选集
        :param word:
        :return:
        """
        a = (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
        return a


if __name__ == "__main__":
    correction1 = Correction(add_weight())
    print(correction1.correction_word("appla"))

