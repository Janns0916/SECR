import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class PreferenceModule(nn.Module):
    def __init__(self, entity_num, concept_num):
        super(PreferenceModule, self).__init__()
        self.bert_model = BertModel.from_pretrained(r'E:\janns\MACR-master2\bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained(r'E:\janns\MACR-master2\bert-base-uncased')
        self.bilinear_layer = nn.Bilinear(entity_num, self.bert_model.config.hidden_size, concept_num)  # 64368*768
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(768, 128)  # 768*128

    def encode_text(self, text):
        inputs = self.bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=30,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(2)
        return embeddings

    def forward(self, context, response, entity_vec, affective_input):
        context_text = ' '.join(context)
        context_encoding = self.encode_text(context_text)  # 1*30*768
        context_embedding = torch.mean(context_encoding, dim=1)  # 1*768

        response_text = ' '.join(response)
        response_encoding = self.encode_text(response_text)  # 1*30*768
        response_embedding = torch.mean(response_encoding, dim=1)  # 1*768

        fused_embedding = torch.cat((context_embedding, response_embedding), dim=0)  # 2*768
        fused_embedding = self.linear(fused_embedding)

        output = self.softmax(fused_embedding)

        return output


def sentiment():
    entity_num = 64368  # 假设实体数量为100
    concept_num = 10  # 假设属性数量为10

    preference_model = PreferenceModule(entity_num, concept_num)
    for data in ['entity_train_data','entity_test_data','entity_valid_data']:
        with open(f'../data/attr/{data}.pkl', 'rb') as f:
            entity_attribute = pickle.load(f)
    for data in ['text_train_data','text_test_data','text_valid_data']:
        with open(f'../data/attr/{data}.pkl', 'rb') as f:
            text_attribute = pickle.load(f)

        entity_vec = torch.zeros(entity_num)
        affective_input_size = 128
        affective_input = torch.zeros(affective_input_size)  # 根据实际情况定义affective_input

    text_output = []
    for key, context in tqdm(entity_attribute.items()):
        for keys, response in text_attribute.items():
            if key == keys:
                output = preference_model(context, response, entity_vec, affective_input)
        # 这个里面存储的是对文本抽取所得到的特征
            text_output.append(output)
    # 通过前三个得到的最高属性信息
    top_3_indices = torch.topk(output, k=3).indices.tolist()
    print(top_3_indices)   #[[76, 3, 116], [76, 105, 115]]
    return text_output, top_3_indices


if __name__ == '__main__':
    sentiment()
