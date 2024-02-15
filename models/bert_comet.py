import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel
from transformers import BertTokenizer, BertForSequenceClassification
from torch import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PreferenceModule(nn.Module):
    def __init__(self, entity_num, concept_num):
        super(PreferenceModule, self).__init__()
        # self.bert_model = BertModel.from_pretrained('../bert-base-uncased')
        self.bert_model = BertModel.from_pretrained(r'E:\Jann\MACR-master\bert-base-uncased')
        # self.bert_tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained(r'E:\Jann\MACR-master\bert-base-uncased')
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


class AttributeSelfAttentionEmbedding(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttributeSelfAttentionEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size)
        self.attention = nn.MultiheadAttention(output_size, num_heads=8)
        self.fc = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Embedding layer
        x = x.unsqueeze(1)  # Add an extra dimension for sequence length
        x, _ = self.attention(x, x, x)  # Self-attention layer
        x = x.squeeze(1)  # Remove the extra sequence length dimension
        x = self.fc(x)  # Linear layer
        return x


def sentiment():
    entity_num = 64368  # 假设实体数量为100
    concept_num = 3  # 假设属性数量为10
    path = '../data/attr/'
    preference_model = PreferenceModule(entity_num, concept_num)
    for data in ['entity_train_data', 'entity_test_data', 'entity_valid_data']:
        # with open(f'../data/attr/{data}.pkl', 'rb') as f:
        with open(rf'E:\Jann\MACR-master\data\attr\{data}.pkl', 'rb') as f:
            entity_attribute = pickle.load(f)

    for data in ['text_train_data', 'text_test_data', 'text_valid_data']:
        # with open(f'../data/attr/{data}.pkl', 'rb') as f:
        with open(rf'E:\Jann\MACR-master\data\attr\{data}.pkl', 'rb') as f:
            text_attribute = pickle.load(f)

        entity_vec = torch.zeros(entity_num)
        affective_input_size = 6766
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
    # print(top_3_indices)   #[[65, 36, 67], [65, 52, 67]]
    return text_output, top_3_indices


def get_result():
    attr_list = []
    text_output, top_3_indices = sentiment()
    for data in top_3_indices:
        attr_list.append(data)
    return attr_list
    # print(attr_list)  # [[65, 97, 7], [65, 46, 8]]
# [65]
# [97, 7, 46, 8]

def get_common():
    attr_list = get_result()

    common = []
    unique = []

    # 创建一个空集合用于存储相同的值
    common_set = set()

    # 遍历attr_list中的子列表
    for sublist in attr_list:
        # 遍历子列表中的元素
        for value in sublist:
            if value in common_set:
                # 如果值已经存在于common_set中，则将其添加到common列表中
                common.append(value)
            else:
                # 如果值不存在于common_set中，则将其添加到common_set和unique列表中
                common_set.add(value)
                unique.append(value)
    return attr_list, common, unique
    print(attr_list)
    print("Common Values:", common)
    print("Unique Values:", unique)


# Common Values: [40]
# Unique Values: [54, 40, 10, 26, 4]


def comet_redial():
    tensor_sentiment_data = {}
    text_feature = pickle.load(open(r'E:\janns\MACR-master3\data\attr\entity_train_data_features_comet.pkl', 'rb'))
    for context in text_feature:
        for k, v in context.items():
            v = torch.tensor(v)
            # 把所有的value转换成tensor类型，然后存储到tensor_sentiment_data
            tensor_sentiment_data[k] = v
    # for k1, v1 in tensor_sentiment_data.items():
    #     print(k1)
    #     print(v1)
    return tensor_sentiment_data


def bert_redial():
    model_name = r'E:\janns\MACR-master3\bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3个情感类别
    data = pickle.load(open(r'E:\janns\MACR-master3\data\attr\text_train_data.pkl', 'rb'))
    # data = {
    #     '3451': ["Well I think I'm looking for a really scary movie, something like Lights Out (2016).",
    #              'Did you see Wrong Turn (2003)?',
    #              "It's a scary movie with Eliza Dushku.",
    #              "Yeah, that was terrifying. Do you have something that's maybe less violent?",
    #              'Something else that scared me was Scream (1996)',
    #              'It is still violent, but campy enough to take the edge off.',
    #              'That was pretty good. Have you seen The Others (2001)?',
    #              'That is a good one too.',
    #              "Okay, I'll think I'll go fall asleep to Scream. Thanks for your suggestions."],
    #     '10344': ['looking for a good spoof movie like Scary Movie (2000) or Airplane! (1980)',
    #               "Oh loved Scary Movie (2000) I haven't seen Airplane! (1980)",
    #               'Have you seen Shaun of the Dead (2004)',
    #               "No, I've heard a lot about it",
    #               'what is it like?',
    #               'It is a zombie spoof I really liked it super funny',
    #               'That sounds great',
    #               'any others?',
    #               'Scary Movie 3 (2003) is a great follow up to the first',
    #               'Ive seen that and loved it',
    #               'Oh or A Haunted House (2013) FUNNY!',
    #               'the cast was amazing in it',
    #               'Never heard of that is it like Scary Movie (2000) ?',
    #               'Yes right along those lines I am sure you would enjoy it',
    #               'awesome thanks!',
    #               'bye']}
    results = {}
    labels_mapping = {0: '积极', 1: '消极', 2: '中性'}  # 情感类别标签映射
    for key, sentences in tqdm(data.items()):
        results[key] = {}
        max_positive_prob = 0.0
        max_negative_prob = 0.0
        max_neutral_prob = 0.0
        max_probility = 0
        for sentence in sentences:
            inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probabilities = torch.softmax(logits, dim=0).tolist()
            predicted_label = torch.argmax(logits).item()

            positive_prob = probabilities[0]
            negative_prob = probabilities[1]
            neutral_prob = probabilities[2]

            if positive_prob > max_positive_prob:
                max_positive_prob = positive_prob
            if negative_prob > max_negative_prob:
                max_negative_prob = negative_prob
            if neutral_prob > max_neutral_prob:
                max_neutral_prob = neutral_prob

            # result = {
            #     'sentence': sentence,
            #     'emotion': predicted_label,
            #     'positive_prob': positive_prob,
            #     'negative_prob': negative_prob,
            #     'neutral_prob': neutral_prob,
            #     'text_encoding': inputs['input_ids'].tolist()
            # }
            result = {
                'key': key,
                'sentence': sentence,
                'emotion': predicted_label,
                'positive_prob': positive_prob,
                'negative_prob': negative_prob,
                'neutral_prob': neutral_prob,
                'max_positive_prob: ': max_positive_prob,
                'max_negative_prob: ': max_negative_prob,
                'max_neutral_prob: ': max_neutral_prob,
                'max_probility: ': max_probility,
                'positive_weight: ': max_positive_prob / (max_positive_prob + max_negative_prob + max_neutral_prob),
                'negative_weight: ': max_negative_prob / (max_positive_prob + max_negative_prob + max_neutral_prob),
                'neutral_weight: ': max_neutral_prob / (max_positive_prob + max_negative_prob + max_neutral_prob),
                'positive_label: ': labels_mapping[
                    torch.argmax(torch.Tensor([max_positive_prob, max_negative_prob, max_neutral_prob])).item()],
                'text_encoding': inputs['input_ids'].tolist()
            }
            results[key][sentence] = result

        results[key]['key'] = key
        results[key]['max_probility'] = max_probility
        results[key]['positive_weight'] = max_positive_prob / (max_positive_prob + max_negative_prob + max_neutral_prob)
        results[key]['negative_weight'] = max_negative_prob / (max_positive_prob + max_negative_prob + max_neutral_prob)
        results[key]['neutral_weight'] = max_neutral_prob / (max_positive_prob + max_negative_prob + max_neutral_prob)
        results[key]['positive_label'] = labels_mapping[
            torch.argmax(torch.Tensor([max_positive_prob, max_negative_prob, max_neutral_prob])).item()]
    # print(results)
    return results


def comet_trans_shape():
    data_dict = comet_redial()

    linear_layer1 = nn.Linear(768, 128)

    # 创建一个空的字典，用于存储转换后的数据
    transformed_data_dict = {}
    n = 128
    # 将每个数据转换为 (128*128) 维度
    for key, data in tqdm(data_dict.items()):
        # 将数据转换为 PyTorch 张量
        input_tensor = torch.tensor(data)

        # 将数据应用于线性层进行维度转换
        output_tensor = linear_layer1(input_tensor)
        output_tensor = output_tensor.repeat(n, 1)
        output_tensor = output_tensor.transpose(1, 0)
        linear_layer2 = nn.Linear(output_tensor.shape[1], n)
        output_tensor = linear_layer2(output_tensor)
        reshaped_tensor = torch.squeeze(output_tensor, dim=0)
        transformed_data_dict[key] = reshaped_tensor  # 里面都是128*128
    return transformed_data_dict


def fusion_comet_bert():
    transformed_data_dict = comet_trans_shape()
    bert_redials = bert_redial()
    for k, v in bert_redials.items():
        positive_weight = v['positive_weight']
        negative_weight = v['negative_weight']
        neutral_weight = v['neutral_weight']
        positive_label = v['positive_label']
        break
    bert_encoder = {'positive_weight': positive_weight, 'negative_weight': negative_weight,
                    'neutral_weight': neutral_weight, 'positive_label': positive_label}
    max_attr_weight = max(bert_encoder['positive_weight'], bert_encoder['negative_weight'],
                          bert_encoder['neutral_weight'])
    final_fusion_data = {}
    for k, v in bert_redials.items():
        for k1, v1 in transformed_data_dict.items():
            try:
                if k == k1:
                    v1 *= max_attr_weight
                    # v1 = F.norm(v1)
                    final_fusion_data[k] = v1
            except:
                pass
    # print(final_fusion_data)
    return final_fusion_data

    # for k, v in final_fusion_data.items():
    #     print("k: ", k, "v.shape:  ", v.shape)


def read_pkl(path):
    res = pickle.load(open(path, 'rb'))
    print(res)
    return res


if __name__ == '__main__':
    path = '../data/attr/attr_dict.pkl'
    # read_pkl(path)
    # fusion_comet_bert()
    sentiment()
