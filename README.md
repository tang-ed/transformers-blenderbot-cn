# transformers-blenderbot-cn
基于transformers的开源Facebook中的blenderbot训练中文聊天机器人
## 安装支持库
pip3 install -r requirements.txt
## 训练步骤
1.python3 new_token.py

2.python3 train.py

## 使用官方预训练模型训练
python3 pre_train.py

## 注意：
非训练是将config.json中的use_cache改为true

若是训练时loss下降缓慢或者很难下降，将loss直接改为keras.losses.SparseCategoricalCrossentropy()，或者将optimizer改为默认的adam

# transformers库的链接

https://huggingface.co/transformers/

# 两个数据样本进行了30个epoch的训练，测试图在image文件中。loss最终下降到了0.04左右。
