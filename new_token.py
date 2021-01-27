from transformers import BlenderbotSmallTokenizer
import numpy as np
from config_ import cfg


token = BlenderbotSmallTokenizer(vocab_file="vocab.json", merges_file="merges.txt")

que = []
ans = []
start_token = token.bos_token
end_token = token.eos_token


with open(cfg.data_file) as f:
    data = f.read().split("\n")


    state = None
    q = None
    a = None

    for i in range(len(data)):
        if data[i] != " " and data[i] != "":
            if data[i + 1] != "" and data[i+1] != " ":
                if q is None:
                    q = data[i]
                    a = data[i+1]
                    que.append(q)
                    ans.append(a)

                    state = q+" | " + a
                    continue
                if state is not None:
                    q = state
                    a = data[i + 1]
                    state = q + " | " + a
                    que.append(q)
                    ans.append(a)


            else:
                continue
        else:
            state = None
            q = None
            a = None

que_data = token(que, return_tensors="tf", padding=True, truncation=True)
ans_data = token(ans, return_tensors="tf", padding=True, truncation=True, max_length=126)
ans_datas = ans_data["input_ids"]
# print(ans_datas)

new_ans = []
for i in ans_datas:
    ans_ls = list(i.numpy())
    ans_ls.insert(0, 1)
    try:
        z_index = ans_ls.index(0)
        ans_ls.insert(z_index, 2)
    except:
        ans_ls.insert(-1, 2)
    new_ans.append(ans_ls)

ans_data = np.array(new_ans)
print(ans_data.shape)
np.savez("train_data", que_data["input_ids"], ans_data)
print("完成")
