import gensim
import numpy as np

# 下载并解压 GloVe 预训练模型，假设我们使用 GloVe.840B.300d.txt
glove_file = "E:\\Glove\\glove.840B.300d.txt"

# 加载 GloVe 模型
import numpy as np

def load_glove_model(glove_file):
    print("Loading GloVe model...")
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                # Try to convert values[1:] to a float vector
                vector = np.array(values[1:], dtype='float32')
                model[word] = vector
            except ValueError:
                # If conversion fails, skip the line
                print(f"Skipping line: {line.strip()}")
                continue
    print("GloVe model loaded successfully.")
    return model


# 加载模型
glove_model = load_glove_model(glove_file)

# 获取单词的词向量
word = "king"
vector = glove_model.get(word)
print(f"Vector for '{word}':", vector)

def get_sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.get(word) for word in words if word in model]
    if len(word_vectors) == 0:
        return np.zeros(100)  # 如果没有找到任何词，返回全零向量
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector

# 示例句子
sentence = "King is powerful"
sentence_vector = get_sentence_vector(sentence, glove_model)
print(f"Sentence vector for '{sentence}':", sentence_vector)

