# transformers-blenderbot-cn
基于transformers的开源Facebook中的blenderbot训练中文聊天机器人
## 安装支持库
pip3 install -r requirements.txt
## 训练步骤
1.python3 create_json.py 创建vocab.json文件

2.python3 train.py

## 测试步骤
1.python3 save_trans_model.py 生成BlenderbotSmall专属的权重文件

2.python3 test.py

## 注意：
非训练是将config_small.json中的use_cache改为true

若是训练时loss下降缓慢或者很难下降，将loss直接改为keras.losses.SparseCategoricalCrossentropy()，或者将optimizer改为默认的adam


还有关于tokenizer的问题，本tokenizer是个人写，适用于中文，英文的话，需要自己改一下。

# transformers库的链接

https://huggingface.co/transformers/

# transformers-blenderbot模型源码
https://huggingface.co/transformers/_modules/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.html#TFBlenderbotSmallModel

# 两千个数据样本进行了30个epoch的训练，测试图在image文件中。loss最终下降到了0.04左右。


