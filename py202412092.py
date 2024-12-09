import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torch.nn as nn

# 加载音频数据
def load_audio(file_path, target_sampling_rate=16000):
    # librosa 自动将音频归一化到 [-1, 1]
    audio, sr = librosa.load(file_path, sr=target_sampling_rate, mono=True)
    return audio, sr

# 替换模型代码
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

# 加载预训练模型和处理器
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

# 音频文件路径
audio_file_path = "This_is_wonderful.wav"

# 加载音频数据
signal, sampling_rate = load_audio(audio_file_path, target_sampling_rate=16000)

# 运行模型
def process_audio(
    signal: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    # 预处理音频
    processed = processor(signal, sampling_rate=sampling_rate)
    input_values = torch.tensor(processed.input_values).to(device)

    # 模型推理
    with torch.no_grad():
        output = model(input_values)[0 if embeddings else 1]

    # 转为 NumPy 格式
    return output.cpu().numpy()

# 获取情感预测值
predicted_emotions = process_audio(signal, sampling_rate)
#唤醒值、支配度、效价值
print("情感预测值:", predicted_emotions)

# 提取嵌入
embeddings = process_audio(signal, sampling_rate, embeddings=True)
print("嵌入特征:", embeddings)
