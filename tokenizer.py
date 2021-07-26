import json
from collections import OrderedDict
from collections import defaultdict
from typing import List, Union
import numpy as np


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            len_x = len(x)
            lengths.append(len_x)
            if flag and len_x:
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


class Tokenizer:
    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split = ' ', special_tokens="__null__"):
        self.filters = [i for i in filters]
        self.split = split
        self.special_tokens = special_tokens

    def fit_on_texts(self, sentences):

        word_dir = OrderedDict()
        for sentence in sentences:
            s = sentence.split(self.split)
            s = [i for i in s if i not in self.filters]
            for n in s:
                if n not in word_dir:
                    word_dir[n] = 1
                else:
                    word_dir[n] += 1
        word_ls = list(word_dir.items())

        word_ls.sort(key=lambda x:x[1], reverse=True)

        if self.special_tokens is None:
            s_tokens = []
            s_tokens.extend([i[0] for i in word_ls])
            self.word_index = dict(zip(s_tokens, range(1, len(s_tokens)+1)))
        else:
            s_tokens = self.special_tokens
            s_tokens.extend([i[0] for i in word_ls])
            self.word_index = dict(zip(s_tokens, range(0, len(s_tokens))))



class SelfTokenizer:
    def __init__(self, vocab_file=None):
        """
        :param vocab_file: 关于vocab的json文件。
        """
        self.vocab_file = vocab_file
        if self.vocab_file is not None:
            self.vocab = self.read_json()

    def read_json(self):
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def train(self, files: Union[str, List[str]]=None, special_tokens: Union[str, List[str]]=None):
        """
        :param files: [file1, file2, file3.....]
        :param special_tokens: ["__null__", "__start__", "__end__", "__unk__", "__newln__"]， 也可以自己指定。
        """
        assert files is not None, "files cannot be None"
        tokenizer = Tokenizer(special_tokens=special_tokens)
        datas = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                data = f.readlines()
                new_data = [" ".join([n for n in i]) for i in data]
                datas.extend(new_data)

        tokenizer.fit_on_texts(datas)
        self.vocab_dir = tokenizer.word_index

        return len(self.vocab_dir)

    def save(self, file):
        """
        :param file: file or file.json
        """
        if ".json" not in file:
            file += ".json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(self.vocab_dir, f, ensure_ascii=False, indent=4)

    def decoder(self, sentence_num, remove_flag=True):
        """
        :param sentence_num: tensorflow的张量数组    维度为2
        :param remove_flag:去除特殊令牌
        """
        vocab = {v:k for k, v in self.vocab.items()}
        sentences = []
        try:
            sentence_num = sentence_num.numpy()
        except:
            sentence_num = sentence_num
        axis = len(sentence_num.shape)
        if axis == 1:
            sentence_num = sentence_num[None, :]
        for sentence in sentence_num:
            if remove_flag:
                new_sentence = [vocab[i] for i in sentence if i not in [0, 1, 2, 3, 4]]
            else:
                new_sentence = [vocab[i] for i in sentence]
            sentences.append("".join(new_sentence))
        return sentences

    def encoder(self, text, add_special_tokens=False, padding=False, truncation=False, max_len=None, return_tensor="np"):
        """
        :param text: [sentence1, sentence2, sentence3.....]
        :param add_special_tokens: 添加开始标志和结束标志。
        :param padding:长度统一
        :param truncation:和padding一样，一起为True
        :param max_len:最大长度
        :param beginning_to_end:剪取的顺序，True为从index为零开始到max_len，False为从index为-1开始到-max_len。
        :param return_tensor:返回的类型，默认是numpy，有tf，pt，pd，np
        """
        vocab = self.vocab
        text_num = []
        for t in text:
            new_t = [n for n in t]
            num_ls = []
            for i in new_t:
                if i == "" or i == " ":
                    continue
                try:
                    num = vocab[i]
                except:
                    num = vocab["__unk__"]
                num_ls.append(num)
            text_num.append(num_ls)

        if padding and truncation and max_len and not add_special_tokens:
            text_num = pad_sequences(text_num, padding="post", truncating="post", maxlen=max_len)

        elif padding and truncation:
            if add_special_tokens and not max_len:
                new_text = []
                for i in text_num:
                    i.append(vocab["__end__"])
                    i.insert(0, vocab["__start__"])
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post")

            elif add_special_tokens and max_len:
                new_text = []
                for i in text_num:
                    if len(i) < max_len - 2:
                        i.append(vocab["__end__"])
                        i.insert(0, vocab["__start__"])
                    else:
                        i = i[:max_len-2]
                        i.append(vocab["__end__"])
                        i.insert(0, vocab["__start__"])
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post", maxlen=max_len)

            elif not add_special_tokens and not max_len:
                new_text = []
                for i in text_num:
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post")

            elif not add_special_tokens and max_len:
                new_text = []
                for i in text_num:
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post", maxlen=max_len)

        else:
            assert len(text_num) == 1, "You should specify padding=True and truncating=True"

            if max_len:
                text_num = [text_num[0][:max_len]]

            if add_special_tokens:
                new_text = text_num[0]
                new_text.append(vocab["__end__"])
                new_text.insert(0, vocab["__start__"])
                text_num = [new_text]
        if return_tensor == "tf":
            import tensorflow as tf
            input_ids = tf.constant(text_num)
            input_act = tf.constant(input_ids.numpy()>0, dtype="int32")
        elif return_tensor == "pd":
            import paddle
            input_ids = paddle.to_tensor(text_num, dtype="int64")
            input_act = paddle.to_tensor(input_ids.numpy() > 0, dtype="int64")
        elif return_tensor == "pt":
            import torch
            input_ids = torch.tensor(text_num)
            input_act = torch.tensor(input_ids.numpy() > 0, dtype="int32")
        else:
            input_ids = np.array(text_num)
            input_act = np.array(input_ids > 0, dtype="int32")
        return {
            "input_ids":input_ids,
            "attention_mask":input_act
        }


if __name__ == '__main__':
    tokenizer = SelfTokenizer("vocab.json")
    inputs = tokenizer.encoder(["你好，我 是 唐小书。 | 好的啊。", "好的。"],padding=True, truncation=True, add_special_tokens=True, max_len=20, return_tensor="pd")
    input_ids = inputs["input_ids"]
    print(input_ids)
    print(tokenizer.decoder(input_ids))
