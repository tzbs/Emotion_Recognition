import torch
from transformers import RobertaTokenizer
import os

# 加载 tokenizer
tokenizer = RobertaTokenizer.from_pretrained('E:\\Emotion_Recognition\\xiji_trian')

# 重新构建模型结构
from transformers import RobertaModel
import torch.nn as nn

class RoBERTaRegressor(nn.Module):
    def __init__(self, roberta_model):
        super(RoBERTaRegressor, self).__init__()
        self.roberta = roberta_model
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 2)  # 输出 V 和 A

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的隐藏状态
        logits = self.regressor(cls_output)  # 输出 V 和 A 的预测值
        return logits

# 重新加载 RoBERTa 模型并加载训练好的权重

roberta = RobertaModel.from_pretrained("E:\\Emotion_Recognition\\roberta_base")
model = RoBERTaRegressor(roberta)
# state_dict=model.load_state_dict(torch.load(os.path.join("E:\\Emotion_Recognition\\xiji_trian", 'roberta_regressor.pth'),map_location=torch.device('cpu'))  # 加载保存的权重
# model.load_state_dict(state_dict, strict=False)


# 使用 strict=False 来忽略不匹配的权重
state_dict = torch.load(os.path.join("E:\\Emotion_Recognition\\xiji_trian", 'roberta_regressor.pth'), map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)


# 将模型加载到设备（GPU 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 设置模型为评估模式
model.eval()

# 推理函数
def predict(text, model, tokenizer):
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
    v_pred = outputs[0, 0].item()  # 第一个输出是 V
    a_pred = outputs[0, 1].item()  # 第二个输出是 A
    return v_pred, a_pred

# 输入文本进行推理
text = "I am very happy today!"
v, a = predict(text, model, tokenizer)
print(f"Predicted V: {v}, Predicted A: {a}")

# 反归一化输出（如果需要得到实际值）
def denormalize(value, min_val, max_val):
    return (value + 1) * (max_val - min_val) / 2 + min_val

# 反归一化 V 和 A
V_MIN, V_MAX = 1.2, 4.6
A_MIN, A_MAX = 1.8, 4.4
v_actual = denormalize(v, V_MIN, V_MAX)
a_actual = denormalize(a, A_MIN, A_MAX)
print(f"Actual V: {v_actual}, Actual A: {a_actual}")

