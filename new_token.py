from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import tensorflow as tf
from config_ import cfg


token = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{}~\t\n',)

que = []
ans = []
start_token = "<start>"
end_token = "<end>"

with open(cfg.data_file) as f:
    data = f.read().split("\n")

    state = None
    q = None
    a = None

    for i in range(len(data)):
        if data[i] != "  " and data[i] != " ":
            if data[i + 1] != "  " and data[i+1] != " ":
                if q is None:
                    q = data[i]
                    a = data[i+1]
                    que.append(start_token + " " + q + " " + end_token)
                    ans.append(start_token + " " + a + " " + end_token)
                    state = q+" | " + a
                    continue
                if state is not None:
                    q = state
                    a = data[i + 1]
                    state = q + " | " + a
                    que.append(start_token + " " + q + " " + end_token)
                    ans.append(start_token + " " + a + " " + end_token)

            else:
                continue
        else:
            state = None
            q = None
            a = None

token.fit_on_texts(que+ans)
vocab_size = len(token.word_index) + 1
print(vocab_size)

que = token.texts_to_sequences(que)
ans = token.texts_to_sequences(ans)
que = pad_sequences(que, padding="post")
ans = pad_sequences(ans, padding="post")

que_mask = np.array(que > 0, dtype="int32")
ans_mask = np.array(ans > 0, dtype="int32")

with open("vocab.json", "w") as f:
    json.dump(token.word_index, f, ensure_ascii=False, indent=4)

with open("config.json" , "r") as f:
    data = json.load(f)

data["vocab_size"] = vocab_size
with open("config.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


np.savez("cn_train_data", que, que_mask, ans, ans_mask)
#
print("保存完成")
