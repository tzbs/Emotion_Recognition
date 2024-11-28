import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from tqdm import tqdm


# 1. 加载本地的 RoBERTa 模型和分词器
local_model_path = "E:\\Emotion_Recognition\\roberta_base"  # 修改为你的实际路径
tokenizer = RobertaTokenizer.from_pretrained(local_model_path)
roberta = RobertaModel.from_pretrained(local_model_path)

# 2. 数据归一化函数
def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1

# 反归一化函数（如果需要）
def denormalize(value, min_val, max_val):
    return (value + 1) * (max_val - min_val) / 2 + min_val

# 3. 加载 EmoBank 数据集，并归一化 V 和 A
V_MIN, V_MAX = 1.2, 4.6
A_MIN, A_MAX = 1.8, 4.4

df = pd.read_csv("emobank_small.csv")  # 修改为你的数据路径
df['V_normalized'] = df['V'].apply(lambda v: normalize(v, V_MIN, V_MAX))
df['A_normalized'] = df['A'].apply(lambda a: normalize(a, A_MIN, A_MAX))

# 4. 自定义数据集类
class EmoBankDataset(Dataset):
    def __init__(self, texts, v_labels, a_labels, tokenizer, max_length=512):
        self.texts = texts
        self.v_labels = v_labels
        self.a_labels = a_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        v_label = self.v_labels[idx]
        a_label = self.a_labels[idx]

        # 分词处理
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "v_label": torch.tensor(v_label, dtype=torch.float),
            "a_label": torch.tensor(a_label, dtype=torch.float),
        }

# 5. 创建训练集、验证集、测试集的 DataLoader
train_dataset = EmoBankDataset(
    texts=df[df['split'] == 'train']['text'].tolist(),
    v_labels=df[df['split'] == 'train']['V_normalized'].tolist(),
    a_labels=df[df['split'] == 'train']['A_normalized'].tolist(),
    tokenizer=tokenizer
)
val_dataset = EmoBankDataset(
    texts=df[df['split'] == 'dev']['text'].tolist(),
    v_labels=df[df['split'] == 'dev']['V_normalized'].tolist(),
    a_labels=df[df['split'] == 'dev']['A_normalized'].tolist(),
    tokenizer=tokenizer
)
test_dataset = EmoBankDataset(
    texts=df[df['split'] == 'test']['text'].tolist(),
    v_labels=df[df['split'] == 'test']['V_normalized'].tolist(),
    a_labels=df[df['split'] == 'test']['A_normalized'].tolist(),
    tokenizer=tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 6. 定义回归模型
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

# 7. 加载模型到设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RoBERTaRegressor(roberta).to(device)

# 8. 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

# 9. 训练模型
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        v_labels = batch["v_label"].to(device)
        a_labels = batch["a_label"].to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(input_ids, attention_mask)
        v_loss = loss_fn(outputs[:, 0], v_labels)
        a_loss = loss_fn(outputs[:, 1], a_labels)
        loss = v_loss + a_loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")


# 保存模型的状态字典 (state_dict)
torch.save(model.state_dict(), 'E:\\Emotion_Recognition\\roberta_regressor.pth')

# 保存 tokenizer
tokenizer.save_pretrained('E:\\Emotion_Recognition')

# 10. 推理函数
def predict(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
    v_pred = outputs[0, 0].item()
    a_pred = outputs[0, 1].item()
    return v_pred, a_pred

# 示例推理
text = "I am very happy today!"
v, a = predict(text, model, tokenizer)
print(f"Predicted V: {v}, Predicted A: {a}")

# 11. 反归一化结果（如果需要实际范围值）
v_actual = denormalize(v, V_MIN, V_MAX)
a_actual = denormalize(a, A_MIN, A_MAX)
print(f"Actual V: {v_actual}, Actual A: {a_actual}")