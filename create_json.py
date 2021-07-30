from tokenizer import SelfTokenizer
import json


tokenizer = SelfTokenizer()


vocab_size = tokenizer.train(["./data/train_data.txt"], special_tokens=["__null__", "__start__", "__end__","|" , "__unk__"])
tokenizer.save("vocab")

with open("model_file/config_small.json", "r") as f:
    data = json.load(f)



data["vocab_size"] = vocab_size

with open("model_file/config_small.json", "w") as fs:
    json.dump(data, fs, ensure_ascii=False, indent=4)

