import torch
from fairseq.models.roberta import RobertaModel

# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('roberta.base', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
# 测试输入文本
input_text = "I am a student"
tokens = roberta.encode(input_text)

# 获取模型的输出
with torch.no_grad():
    features = roberta.extract_features(tokens)  # 或者roberta.forward(tokens)

# 打印模型输出的特征
print(features)






