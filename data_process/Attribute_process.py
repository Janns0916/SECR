"""
现在要做的一个工作是根据数据集将一个类别的属性拿出来，为其进行编号
然后根据对话中提到的id去找对应的属性，找到对应的属性后，我们将他们添加到实体中去。
"""
# coding:utf-8
import os
import pickle

from tqdm import tqdm

from model_copy_helpful_ntrd import concept_edge_list4GCN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models.transformer import TorchGeneratorModel, _build_encoder, _build_decoder, _build_encoder_mask, \
    _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings, _create_entity_embeddings
from models.graph import SelfAttentionLayer, SelfAttentionLayer_batch
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import json
from models import bert_comet
from models.bert_comet import AttributeSelfAttentionEmbedding


def attr_change_id():
    # 存储导演信息
    director = []
    directors = {}

    # 存储制片人信息
    produced_by = []
    produces_by = {}

    # 存储主演信息
    starring = {}
    stars = []

    # 存储语言信息
    language = []
    languages = {}

    # 存储作者信息
    writer_by = []
    writers_by = {}

    # 打开文件
    with open('../data/lmkg/movie_info2.jsonl', 'r', encoding="utf-8") as f:
        for index, line in enumerate(f):
            # json加载数据
            data = json.loads(line)
            # 读取里面详细的信息
            Starring = data['Starring']
            Directed_by = data['Directed_by']
            Produced_by = data['Produced_by']
            Language = data['Language']
            Writer_by = data['Writer_by']
            # 存储主演信息
            for star in Starring:
                stars.append(star)
            # 存储导演信息
            for director_s in Directed_by:
                director.append(director_s)
            # 存储制片人信息
            for product_by in Produced_by:
                produced_by.append(product_by)
            # 存储语言信息
            for language_s in Language:
                language.append(language_s)
            for write in Writer_by:
                writer_by.append(write)

        stars = list(set(stars))
        director = list(set(director))
        produced_by = list(set(produced_by))
        language = list(set(language))
        writer_by = list(set(writer_by))

        for index, star in enumerate(stars):
            starring[index] = star

        for index, director in enumerate(director):
            directors[index] = director

        for index, produced in enumerate(produced_by):
            produces_by[index] = produced

        for index, language in enumerate(language):
            languages[index] = language

        for index, director in enumerate(writer_by):
            writers_by[index] = director
        print(writers_by)
        print(starring)
        print(directors)
        print(produces_by)
        print(languages)
        return starring, directors, produces_by, languages, writers_by


def combine():
    starring, directors, produces_by, languages, writer_by, = attr_change_id()
    star_by_flipped = {v: k for k, v in starring.items()}
    director_by_flipped = {v: k for k, v in directors.items()}
    produce_by_flipped = {v: k for k, v in produces_by.items()}
    language_by_flipped = {v: k for k, v in languages.items()}
    writer_by_flipped = {v: k for k, v in writer_by.items()}

    # 读取inform2.jsonl文件
    with open('../data/lmkg/movie_info2.jsonl', 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
        new_dict = {}
        attr_dict = {}
        # 转换Writer_by列表中的值
        converted_star_by = []
        converted_director_by = []
        converted_produce_by = []
        converted_language_by = []
        converted_writer_by = []
        # 遍历每行数据
        for line in data:
            # 解析JSON数据
            movie_data = json.loads(line)

            # 获取Writer_by列表
            movieId = movie_data['movieId']
            Starring = movie_data['Starring']
            Directed_by = movie_data['Directed_by']
            Produced_by = movie_data['Produced_by']
            Language = movie_data['Language']
            Writer_by = movie_data['Writer_by']

            for star_id in Starring:
                converted_star_by.append(star_by_flipped[star_id])
            new_dict['Starring'] = converted_star_by
            for director_id in Directed_by:
                converted_director_by.append(director_by_flipped[director_id])
            new_dict['Directed_by'] = converted_director_by
            for produce_id in Produced_by:
                converted_produce_by.append(produce_by_flipped[produce_id])
            new_dict['Produced_by'] = converted_produce_by
            for language_id in Language:
                converted_language_by.append(language_by_flipped[language_id])
            new_dict['Language'] = converted_language_by
            for writer_id in Writer_by:
                converted_writer_by.append(writer_by_flipped[writer_id])
            new_dict['Writer_by'] = converted_writer_by
            new_dict['movieId'] = movieId
            attr_dict[new_dict['movieId']] = {'Starring': converted_star_by, 'Directed_by': converted_director_by,
                                                  'Produced_by': converted_produce_by,
                                                  'Language': converted_language_by, 'Writer_by': converted_writer_by}
            print(attr_dict)
            return attr_dict


def attr_change_id2dict():

    starring, directors, produces_by, languages, writer_by = attr_change_id()
    star_by_flipped = {v: k for k, v in starring.items()}
    director_by_flipped = {v: k for k, v in directors.items()}
    produce_by_flipped = {v: k for k, v in produces_by.items()}
    language_by_flipped = {v: k for k, v in languages.items()}
    writer_by_flipped = {v: k for k, v in writer_by.items()}

    # 读取inform2.jsonl文件
    with open('../data/lmkg/movie_info2.jsonl', 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
        attr_dict = {}

        # 遍历每行数据
        for line in data:
            # 解析JSON数据
            movie_data = json.loads(line)

            new_dict = {}
            converted_star_by = []
            converted_director_by = []
            converted_produce_by = []
            converted_language_by = []
            converted_writer_by = []

            # 获取电影属性
            movieId = movie_data['movieId']
            Starring = movie_data['Starring']
            Directed_by = movie_data['Directed_by']
            Produced_by = movie_data['Produced_by']
            Language = movie_data['Language']
            Writer_by = movie_data['Writer_by']

            # 转换属性值为对应的id
            for star_id in Starring:
                converted_star_by.append(star_by_flipped[star_id])
            new_dict['Starring'] = converted_star_by

            for director_id in Directed_by:
                converted_director_by.append(director_by_flipped[director_id])
            new_dict['Directed_by'] = converted_director_by

            for produce_id in Produced_by:
                converted_produce_by.append(produce_by_flipped[produce_id])
            new_dict['Produced_by'] = converted_produce_by

            for language_id in Language:
                converted_language_by.append(language_by_flipped[language_id])
            new_dict['Language'] = converted_language_by

            for writer_id in Writer_by:
                converted_writer_by.append(writer_by_flipped[writer_id])
            new_dict['Writer_by'] = converted_writer_by

            new_dict['movieId'] = movieId
            attr_dict[new_dict['movieId']] = {'Starring': converted_star_by, 'Directed_by': converted_director_by,
                                              'Produced_by': converted_produce_by,
                                              'Language': converted_language_by, 'Writer_by': converted_writer_by}

        return attr_dict


def read_pkl(path):
    res = pickle.load(open(path, 'rb'))
    print(res)

    # print(res.shape)
    # for k, v in res.items():
    #     if k == 20751:
    #         print(k)
    #         print(v)
    #     break
    return res


def save_dict_as_pickle(data, file_path):
    """
    保存字典为.pkl文件
    :param data: 要保存的字典数据
    :param file_path: 文件保存路径
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def read():
    with open('../data/genre/entityId2genre_dict.pkl', 'rb') as f:
        dict_entityId2genre = pkl.load(f)
    genre = dict_entityId2genre
    # print(genre)  # one-hot
    with open('../data/genre/genreId2entityId.pkl', 'rb') as f:
        genreId2entityId = pkl.load(f)
    genreId2entityId = genreId2entityId
    print(genreId2entityId)


def extract_text_from_redial():

    # 读取train.jsonl文件
    with open('../data/dataset/train_data.jsonl', 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    text_dict = {}

    # 遍历数据集
    for item in data:
        conversation_id = item['conversationId']
        messages = item['messages']
        movie_mentions = item['movieMentions']
        text_list = []
        # 提取每个消息的文本
        for message in messages:
            text = message['text']

            # 将@+数字替换为对应的电影名
            # for movie_id, movie_name in movie_mentions.items():
            #     if movie_id in text:
            #         text = text.replace('@' + movie_id, movie_name)
            if movie_mentions == []:
                pass
            else:
                for movie_id, movie_name in movie_mentions.items():
                    if movie_id in text:
                        text = text.replace(f'@{movie_id}', movie_name)
                text_list.append(text)

        # 将对话ID和文本列表存储为字典
        text_dict[conversation_id] = text_list

    # 保存为.pkl文件
    with open('../data/attr/train_data.pkl', 'wb') as file:
        tqdm(pickle.dump(text_dict, file))


def extract_entity_from_redial():
    # 读取train.jsonl文件
    for data in ['train_data','test_data','valid_data']:
        with open(f'data/dataset/{data}.jsonl', 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]

        entity_text_dict = {}
        text_dict = {}
        # 遍历数据集
        for item in data:
            conversation_id = item['conversationId']
            messages = item['messages']
            movie_mentions = item['movieMentions']

            entity_text_list = []
            text_list = []

            # 提取包含@符号的句子，并替换为对应的电影名称
            for message in messages:
                text = message['text']
                if '@' in text:
                    if movie_mentions != []:
                        for movie_id, movie_name in movie_mentions.items():
                            if f'@{movie_id}' in text:
                                text = text.replace(f'@{movie_id}', movie_name)
                        entity_text_list.append(text)
                else:
                    text_list.append(text)
            # 将对话ID和提取的句子列表存储为字典
            entity_text_dict[conversation_id] = entity_text_list
            text_dict[conversation_id] = text_list
        # 保存为.pkl文件
        for data in ['train_data','test_data','valid_data']:
            with open(f'data/attr/entity_{data}.pkl', 'wb') as file:
                tqdm(pickle.dump(entity_text_dict, file))
        for data in ['train_data','test_data','valid_data']:
            with open(f'data/attr/text_{data}.pkl', 'wb') as file:
                tqdm(pickle.dump(text_dict, file))


def demo():
    seed_sets = [[], [], [], [20751, 48522], [20751, 44573, 48522], [20751, 44573, 48522, 53010], [], [44749, 54085, 54441],
     [44749, 54085, 54441], [44749, 54085, 54441], [], [26430, 41923, 55345], [6605, 26430, 41923, 49411, 55345],
     [6605, 26430, 41923, 49411, 55345], [6605, 9021, 22539, 26430, 41923, 49411, 55345], [], [1188, 29031],
     [1188, 29031], [1188, 2069, 29031], [1188, 2069, 29031], [], [13771], [13771, 42372], [287, 13771, 42372],
     [287, 9308, 13771, 42372], [], [29517], [29517], [7374, 29517], [], [3659, 61316], [3659, 61316], [3659, 61316],
     [3659, 61316], [3659, 12995, 61316], [3659, 12995, 61316], [3659, 12995, 52253, 61316],
     [3659, 12995, 52253, 61316], [10034, 49977], [7541, 10034, 44362, 49977], [7541, 10034, 44362, 49977],
     [4872, 7541, 10034, 44362, 49977], [4872, 7541, 10034, 44362, 49977], [4872, 7541, 10034, 44362, 49977], [],
     [8053], [8053, 8576, 9728, 61449], [8053, 8576, 9728, 61449], [], [53114, 63437], [53114, 63437],
     [19266, 53114, 63437], [19266, 53114, 63437], [], [43864, 48382], [43864, 48382], [43864, 48382, 50008],
     [43864, 48382, 50008, 52769], [43864, 48382, 50008, 52769], [8218, 43864, 48382, 50008, 52769], [], [60678],
     [60678], [47380, 60678], [], [], [55949], [55949], [19895, 55949, 57455],
     [19895, 20509, 27115, 38274, 55949, 57455], [19895, 20509, 27115, 38274, 55907, 55949, 57455],
     [894, 19895, 20509, 26430, 27115, 38274, 55907, 55949, 57455],
     [894, 19895, 20509, 26430, 27115, 38274, 55907, 55949, 57455], [9749, 51644, 54320], [9749, 51644, 54320],
     [9749, 26717, 36609, 51644, 54320], [9749, 26717, 36609, 51644, 54320, 54912], [], [32884], [32884],
     [32884, 32886], [32884, 32886, 49719], [32884, 32886, 34061, 49719, 54912], [32884, 32886, 34061, 49719, 54912],
     [], [], [], [], [], [20014, 33160], [20014, 33160], [20014, 24718, 33160], [20014, 24718, 27510, 33160],
     [20014, 24718, 27510, 33160], [], [18840], [12554, 12635, 18840], [10584, 12554, 12635, 18840],
     [10292, 10584, 12554, 12635, 18840, 47569], [], [49294], [49294, 52987], [29640, 49294, 52987],
     [29640, 49294, 52987], [4725, 29640, 49294, 52987], [], [25023], [25023, 54907, 59884],
     [25023, 44224, 54907, 59884], [25023, 44224, 54907, 59884], [21698, 25023, 44224, 54907, 57455, 59884],
     [21698, 25023, 44224, 54907, 57455, 59884], [], [], [34296, 60569], [34296, 60569], [9746, 29527, 34296, 60569],
     [8346, 9746, 29527, 34296, 60569, 62488], [8346, 9746, 29527, 34296, 60569, 62488], [], [7572, 27510],
     [7572, 9552, 27510, 61113], [7572, 9552, 11763, 13561, 24850, 27510, 32906, 48365, 61113], [30598, 30708],
     [30598, 30708], [30598, 30708, 58459], [30598, 30708, 58459], [30598, 30708, 55529, 58459]]
    entity2entityId = pkl.load(open('../data/dbmg/entity2entityId.pkl', 'rb'))
    id2entity = pkl.load(open('../data/dbmg/id2entity.pkl', 'rb'))
    with open('../data/attr/attr_dict.pkl', 'rb') as f:
        attribute = pkl.load(f)
    for seed_set in seed_sets:
        if seed_set == []:
            pass
        for seed in seed_set:
            entity_id = id2entity[seed]
            entity = entity2entityId[entity_id]
            print(entity)


def test():
    with open('../data/attr/attr_dict.pkl', 'rb') as f:
        dict_attr = pkl.load(f)
    db_movie_genre_emb = torch.zeros(128, 128)

    # 情感COMET使用
    attr_list, common, unique = bert_comet.get_common()
    attribute_embedding = AttributeSelfAttentionEmbedding(input_size=128, output_size=128)
    self_attn_db = SelfAttentionLayer(128, 128)

    common = torch.tensor(common)  # (1,)
    unique = torch.tensor(unique)  # (5,)
    commons = attribute_embedding(common)  # (1,128)
    uniques = attribute_embedding(unique)  # (5,128)
    common_representation = self_attn_db(commons)  # (128,)
    unique_representation = self_attn_db(uniques)  # (128,)
    fusions = common_representation + unique_representation  # (128,)
    attr_fusion = torch.mm(fusions.unsqueeze(0).T,
                           fusions.unsqueeze(0))  # (128,128)
    print(attr_fusion.shape)  # (128,128)
    print(db_movie_genre_emb.shape)  # (128,128)


def comet_shape():
    arrays =[1.89202040e-01, 7.49970973e-01, -2.74373703e-02, -1.23444438e+00,
           -8.16708356e-02, -3.37657601e-01, 5.49074411e-02, 8.68136808e-02,
           -1.48138595e+00, 3.78932416e-01, -8.39037716e-01, -2.91084528e-01,
           -2.71413296e-01, 3.59713495e-01, 3.14878702e-01, -3.48715246e-01,
           1.22385156e+00, -1.76470101e-01, -6.56317532e-01, 6.12721920e-01,
           1.28707051e-01, -5.04004359e-02, -8.09759498e-02, -5.70388883e-02,
           -3.90271366e-01, 2.59818584e-01, -1.18859582e-01, 7.67912626e-01,
           -5.47437668e-01, 2.45734215e-01, 8.77644420e-01, -1.08777297e+00,
           -4.12033461e-02, -2.21376717e-01, -3.62692356e-01, 1.13632584e+00,
           3.83431435e-01, -4.92726177e-01, 1.15099296e-01, 9.59111452e-02,
           -1.25427949e+00, 3.29451650e-01, 4.70897257e-01, -1.89009988e+00,
           -4.54304516e-02, -9.15206447e-02, -3.18566203e-01, -4.21631068e-01,
           8.95499766e-01, -4.53700304e-01, 1.60242105e+00, -1.14631712e-01,
           -4.45683181e-01, -3.49802017e-01, -3.74502987e-01, 2.70860612e-01,
           1.02045810e+00, 8.10639799e-01, -2.56297737e-01, 3.71251881e-01,
           7.28112981e-02, 4.82994914e-01, 2.26363391e-01, 6.48675680e-01,
           -6.62558973e-01, 3.52272570e-01, -3.61414433e-01, -8.67265821e-01,
           -2.46196747e-01, 3.32345009e-01, -1.49231553e-02, -5.09762406e-01,
           4.92874622e-01, -2.02772096e-02, 1.03107226e+00, -6.29227042e-01,
           4.44036603e-01, 1.24627560e-01, -8.92523080e-02, -3.87822658e-01,
           6.47551596e-01, 1.64948106e-02, 4.46585745e-01, -2.93133795e-01,
           1.36475176e-01, -6.00850880e-01, 5.86119592e-01, 2.48464614e-01,
           1.93802416e-02, 1.50250256e-01, 1.28911877e+00, 3.60446841e-01,
           -8.24320853e-01, -1.89937517e-01, 3.44655961e-01, -1.13010073e+00,
           3.34231675e-01, -1.65204197e-01, 8.64419222e-01, -2.00404540e-01,
           -2.50809580e-01, 2.63395488e-01, -4.32110101e-01, -7.73985982e-02,
           1.08161962e+00, 6.72041357e-01, 9.48825896e-01, -1.92495257e-01,
           2.34475315e-01, 2.70670623e-01, -1.14874795e-01, -4.09752399e-01,
           -5.94995767e-02, 5.95725298e-01, 2.37429328e-02, -2.65641361e-02,
           -3.76127839e-01, -1.88567042e-02, 7.57568777e-02, -2.01040328e-01,
           2.46867314e-01, -4.37312946e-02, -5.78714572e-02, -2.70222247e-01,
           -3.46042633e-01, 1.17473984e+00, -2.67432660e-01, 4.52141434e-01,
           4.98914123e-01, -3.32624137e-01, 7.76897848e-01, 3.47870886e-01,
           3.04950058e-01, -2.43214369e-01, -1.73642308e-01, -5.35979867e-04,
           -1.66159853e-01, 7.20608592e-01, -1.74972117e-01, -6.77100658e-01,
           4.10591483e-01, -4.65590209e-02, -5.36391810e-02, 1.51114538e-01,
           -1.34658366e-01, -4.77752574e-02, -5.81771433e-01, 7.83501983e-01,
           5.86681604e-01, -1.37300089e-01, -3.47479820e-01, 1.53674901e-01,
           -3.90559286e-02, 2.94537216e-01, 1.77808940e-01, 2.95983136e-01,
           -1.63853317e-01, 9.47746992e-01, 4.99254972e-01, 6.38062835e-01,
           -6.91175997e-01, -7.76982456e-02, -6.90681815e-01, -2.27791220e-02,
           -7.94564709e-02, 1.08016741e+00, 1.78627461e-01, 1.26356542e-01,
           1.10143721e-02, -1.24928629e+00, 4.85935360e-02, -9.18534696e-01,
           -1.28935981e+00, -4.85821664e-01, -6.14192963e-01, -2.16978401e-01,
           3.90896618e-01, 8.99753124e-02, 1.56504959e-01, -1.11261940e+00,
           1.53469443e-02, 1.12614967e-01, -2.27254942e-01, 1.21750258e-01,
           -3.32790554e-01, -6.32094517e-02, 2.59670943e-01, -4.61920574e-02,
           -2.23511145e-01, -1.06402040e-02, -3.33818257e-01, -3.49901989e-02,
           -2.17046663e-01, 1.20562539e-01, -1.90943480e-04, 4.61656451e-02,
           1.85024038e-01, 5.94757557e-01, -7.16075301e-01, -8.84895325e-01,
           7.10797369e-01, -1.21651068e-01, -9.96968985e-01, -6.67748600e-02,
           4.26161438e-01, -5.53059697e-01, -2.57825673e-01, -2.00734690e-01,
           8.77821863e-01, 9.58157837e-01, -4.16547120e-01, -3.69338840e-02,
           5.87269425e-01, 5.15525818e-01, 4.00063753e-01, 1.01862335e+00,
           -3.42683792e-02, -3.11919570e-01, 9.72608566e-01, 5.60292184e-01,
           1.08957803e+00, -8.13524127e-02, 1.97221652e-01, 4.22773249e-02,
           5.98169327e-01, -8.13216865e-01, -8.44480872e-01, 2.16200158e-01,
           -4.36126947e-01, 4.20469224e-01, -7.50992954e-01, -2.86222339e-01,
           -2.96325713e-01, -2.56735533e-02, -2.55237192e-01, 1.33872807e-01,
           -3.77385080e-01, -1.47986352e-01, -7.44628072e-01, -1.33759940e+00,
           9.66206044e-02, -7.48376310e-01, -3.74650300e-01, 2.67952263e-01,
           1.01285525e-01, 2.56474018e-01, 3.24133098e-01, -6.47681892e-01,
           -2.72460319e-02, -7.68338740e-01, 1.05748847e-01, 2.59980798e-01,
           3.42343748e-01, -3.00570130e-02, 5.91848135e-01, -1.10069227e+00,
           -9.15853083e-01, 7.24116325e-01, -8.40004206e-01, -1.16847932e-01,
           1.40845644e+00, 4.08151329e-01, 1.67456269e-02, -5.21419719e-02,
           3.30720335e-01, 1.93973362e-01, -2.57873327e-01, 8.64940405e-01,
           2.09049195e-01, -2.37928689e-01, -2.79582143e-01, -1.88336849e-01,
           3.18735003e-01, -1.09572090e-01, 5.11760652e-01, 8.27786386e-01,
           -7.24938035e-01, 5.71516871e-01, 4.17718351e-01, 2.05975562e-01,
           -3.32655370e-01, 7.11480856e-01, -8.24173629e-01, 5.39123893e-01,
           -4.50606644e-01, -5.19249141e-01, 6.61157429e-01, 1.08664632e-02,
           -1.15734351e+00, 9.63137627e-01, 6.90360904e-01, -4.04094070e-01,
           -7.29956269e-01, 2.16620862e-01, -2.27165088e-01, 4.83109206e-01,
           -1.45151705e-01, -1.32830143e-01, 3.85070920e+00, -4.22657013e-01,
           -7.28442967e-01, 1.22466004e+00, 6.01627864e-03, -5.34357250e-01,
           -6.47390723e-01, 1.08500695e+00, 6.22088730e-01, 4.82369125e-01,
           2.46610910e-01, -7.10676432e-01, 1.47294581e+00, -4.70747232e-01,
           -4.62555699e-02, -3.68386596e-01, -8.44129771e-02, -3.79285216e-01,
           1.46108285e-01, 1.19137034e-01, 3.15165758e-01, -6.33894652e-03,
           1.33713290e-01, 2.31196791e-01, -4.72042799e-01, 1.24303889e+00,
           5.56382611e-02, -2.07277983e-01, 3.32266033e-01, 5.17101228e-01,
           -1.11482069e-01, -7.58563101e-01, -4.63887751e-01, -4.92069423e-01,
           8.83188009e-01, -6.23344183e-02, 1.15416005e-01, -3.98132503e-01,
           -9.19964194e-01, 7.59621382e-01, -3.88119131e-01, 5.91174304e-01,
           -1.26976693e+00, 1.02594122e-02, 1.74583390e-01, -8.86988640e-02,
           5.79309464e-01, -4.91849631e-01, -3.15722644e-01, -6.45931125e-01,
           4.38332468e-01, -2.88449615e-01, -6.60207868e-03, -6.32258892e-01,
           -6.29526436e-01, -1.90467849e-01, 5.64750969e-01, -1.55661121e-01,
           -3.23387414e-01, -6.56594992e-01, -3.88317525e-01, -7.61600196e-01,
           7.37334669e-01, -8.48812521e-01, 1.28609705e+00, -7.37254739e-01,
           2.70287752e-01, 1.48055470e+00, -7.92430401e-01, 2.86277533e-01,
           -1.26271236e+00, -8.72180820e-01, -7.32043982e-02, -7.69209743e-01,
           3.56238708e-02, -5.40244818e-01, 1.81412339e-01, -3.66353840e-02,
           -1.31170556e-01, 5.81593752e-01, -1.45859495e-01, 6.33626357e-02,
           -3.50165105e+00, 1.35792583e-01, 7.64620125e-01, 8.73556137e-02,
           1.08653629e+00, -3.46186697e-01, -9.11652297e-02, -2.58948475e-01,
           -7.15992391e-01, 3.37606132e-01, 6.59981489e-01, -5.92260733e-02,
           -4.29229401e-02, 1.05250403e-01, -4.72905844e-01, -6.85335934e-01,
           -1.00222602e-02, -1.12260532e+00, 9.90820900e-02, -1.44968435e-01,
           -9.16849911e-01, -3.96950245e-01, -1.22661583e-01, -1.03942499e-01,
           2.69991398e-01, -7.46873200e-01, 1.16296008e-01, -1.24853060e-01,
           5.69998562e-01, -6.92531884e-01, 8.83871764e-02, -2.69514680e-01,
           -2.54967451e-01, 1.03838444e+00, -2.68984437e-01, -2.47597694e-01,
           3.58202398e-01, -3.07482872e-02, -2.38041773e-01, 3.27840507e-01,
           4.40989286e-01, 1.12688497e-01, -3.23372900e-01, -4.08968449e-01,
           9.07238126e-02, -1.83361918e-01, 5.99188805e-01, -7.11717129e-01,
           -2.19951808e-01, -1.17994100e-01, -7.53444970e-01, -4.92207557e-02,
           1.11156404e-01, 2.28426546e-01, -5.15684366e-01, -3.66869301e-01,
           -2.57698834e-01, 9.53337252e-02, -2.46281385e-01, 1.54747331e+00,
           -1.08189210e-02, 1.41168892e-01, 2.98050940e-01, -1.65594041e-01,
           -6.78702593e-01, -2.92326868e-01, -1.74804151e-01, 1.25595361e-01,
           -2.47377560e-01, -2.63620406e-01, -9.95205492e-02, -1.15136579e-01,
           1.48557162e+00, -4.37948555e-02, -1.91751644e-01, 1.80971909e+00,
           -9.92200255e-01, 2.92123020e-01, 6.47482425e-02, 7.67044798e-02,
           3.20399463e-01, -1.11468518e+00, 2.41107062e-01, 4.32347924e-01,
           4.13696140e-01, 2.28506699e-01, -1.77804828e-01, 1.28197205e+00,
           -1.29768145e+00, 5.43785870e-01, -1.47696674e-01, 7.04245389e-01,
           1.58340156e-01, 1.71984330e-01, -1.10824931e+00, 8.50665092e-01,
           -3.80289048e-01, 3.24903131e-01, 1.87487036e-01, -3.89630616e-01,
           -9.35420454e-01, -1.05902004e+00, 1.27844065e-01, -1.25251397e-01,
           -9.23481658e-02, 3.20950806e-01, -3.90088111e-01, -1.00011361e+00,
           1.48348227e-01, 2.05747068e-01, 4.44434464e-01, -5.66472560e-02,
           5.25392711e-01, -1.39618844e-01, -4.92439419e-02, 6.25861064e-02,
           6.84550345e-01, -5.00821471e-01, -1.91179276e-01, 5.66658616e-01,
           -5.65457165e-01, -6.75891638e-01, 3.79430920e-01, -2.06179738e-01,
           -1.69366971e-01, 1.45491809e-01, 1.33674353e-01, -6.74814954e-02,
           -2.05301970e-01, 5.14558315e-01, -9.37485278e-01, 4.81763959e-01,
           3.50217894e-02, -4.05758530e-01, -6.13278568e-01, 1.02598429e+00,
           -7.45918334e-01, -7.38744080e-01, -1.61876023e-01, 1.01888955e+00,
           1.74122229e-02, -8.72703612e-01, 6.78797603e-01, 1.49410397e-01,
           7.15816438e-01, 4.32223141e-01, -6.04384959e-01, -4.17851716e-01,
           3.79773349e-01, 3.15154120e-02, 1.01552343e+00, 1.68427825e-04,
           1.00745320e+00, 5.48534751e-01, -5.53326309e-01, 5.57903126e-02,
           4.36866969e-01, -8.22774291e-01, 1.76065117e-02, -3.51549238e-01,
           8.04304838e-01, 4.56699252e-01, 6.49485141e-02, -3.54566455e-01,
           3.95036012e-01, 1.35962144e-02, 5.29266655e-01, 6.88664019e-01,
           1.18107438e-01, 1.49684632e+00, 5.02945542e-01, 3.86393607e-01,
           7.27913380e-01, -1.81404784e-01, 1.49296868e+00, 1.23031449e+00,
           4.95123267e-01, 7.00405762e-02, 5.21773219e-01, 4.13719267e-01,
           8.92919898e-01, -8.99013937e-01, 2.32204888e-02, 1.63981259e-01,
           4.64380145e-01, -4.27274764e-01, -9.03568119e-02, -3.72971386e-01,
           -3.79285365e-02, 6.50530159e-02, -6.75645769e-01, 5.93601108e-01,
           1.05268039e-01, -6.78664073e-02, -1.05682564e+00, 4.74766701e-01,
           -4.29214954e-01, 8.02519858e-01, 7.96036497e-02, -2.38045290e-01,
           -2.13747397e-01, -8.59526694e-01, 8.65920335e-02, 3.85086507e-01,
           5.41912317e-01, 5.00639975e-01, -1.55165300e-01, 1.08739948e+00,
           -1.41273737e+00, 3.34341556e-01, -5.40929914e-01, -9.49016362e-02,
           4.08834547e-01, 2.28283912e-01, 1.25149310e-01, -3.31163377e-01,
           -1.48990214e+00, 2.88738877e-01, 8.73091102e-01, -2.24478990e-01,
           -3.05497766e-01, 1.19419622e+00, -4.87084612e-02, -8.33721280e-01,
           3.35869461e-01, 2.43674397e-01, 3.96645069e-02, 4.28933620e-01,
           5.68319559e-01, 4.12630826e-01, 4.76387501e-01, 3.20432991e-01,
           4.99243408e-01, 8.97436082e-01, -5.45091808e-01, -3.01774830e-01,
           -3.02388281e-01, 5.15237451e-01, -6.79253042e-01, -1.19473505e+00,
           3.43624800e-01, -3.33313435e-01, 3.85844886e-01, -1.10156335e-01,
           -6.32898510e-01, 8.56076404e-02, -7.67668903e-01, -2.43648142e-02,
           4.28352296e-01, -8.99058282e-01, -9.36316073e-01, 6.67405248e-01,
           -7.24455059e-01, -3.19885105e-01, 8.71367693e-01, 1.03077561e-01,
           5.23492277e-01, -3.33210975e-01, -1.63514555e+00, -5.42331576e-01,
           5.54684587e-02, -2.13582903e-01, -3.58788490e-01, -2.15771139e-01,
           7.66717553e-01, 9.13532525e-02, 6.25824094e-01, -1.57223225e-01,
           -3.36902261e-01, -1.36322379e-01, -6.93179250e-01, 7.30593145e-01,
           8.01807821e-01, 2.53670096e-01, 8.01852226e-01, -1.45832181e-01,
           3.25811744e-01, 1.86605275e-01, 9.63828713e-02, 4.21203524e-01,
           -4.19305593e-01, 3.38999152e-01, 7.99167871e-01, -6.99259162e-01,
           -3.21583867e-01, 1.10518716e-01, 9.42526698e-01, 1.74575746e-01,
           -6.74064875e-01, 4.79878396e-01, -1.23808488e-01, -1.39300120e+00,
           -2.21499920e-01, 1.69673383e-01, -1.26003087e-01, -4.39961344e-01,
           9.20217752e-01, 5.02400517e-01, 5.01442730e-01, 7.14476705e-02,
           4.73721981e-01, -6.19212806e-01, 3.19091767e-01, -2.20043182e-01,
           -7.28425145e-01, -3.88122201e-01, -3.32514375e-01, -8.41901183e-01,
           7.33512998e-01, -2.55257875e-01, -3.04747283e-01, -4.64020342e-01,
           4.68328834e-01, 4.76593256e-01, -3.83205712e-01, -6.84076369e-01,
           -2.50880718e-02, 1.49911523e-01, -3.90082777e-01, -9.13188338e-01,
           -4.13776219e-01, 4.97424513e-01, 1.22858874e-01, 5.00781238e-01,
           1.74935198e+00, -1.85360596e-01, 6.46197051e-02, 2.32043803e-01,
           1.75526500e-01, -3.58683765e-02, -1.20214176e+00, -8.32081199e-01,
           -6.39744043e-01, 3.35167170e-01, -9.14519310e-01, -6.70790672e-01,
           -9.09903407e-01, -5.68422735e-01, 1.62068576e-01, -1.01464200e+00,
           2.44579762e-01, -8.01035881e-01, -9.03997064e-01, -7.90875673e-01,
           -6.21605992e-01, 5.16020358e-01, 8.82263958e-01, -5.36288321e-02,
           -7.61125162e-02, 8.20529521e-01, 4.12826896e-01, 1.74412966e-01,
           3.52880239e-01, -9.62465644e-01, 6.99564636e-01, 4.28117752e-01,
           -6.32668197e-01, 4.11262542e-01, 2.36997724e-01, 2.54365325e-01,
           -1.42099068e-01, -5.98942637e-01, 3.46777409e-01, 1.52290151e-01,
           -5.82801923e-02, 4.56885666e-01, -1.09685091e-02, -1.11778185e-01,
           3.41766179e-01, -2.39650086e-01, -6.45759050e-03, 2.61454582e-01,
           1.25424862e+00, 7.30980337e-01, -6.06960595e-01, 5.06115198e-01,
           -1.43972969e+00, -1.75456882e-01, 1.40798643e-01, 8.47667456e-01,
           -3.28709602e-01, -2.49079585e-01, 6.07924581e-01, 2.65605330e-01,
           -5.83583534e-01, -6.68025374e-01, -5.83986491e-02, 6.22198582e-01]
    print(len(arrays))


if __name__ == '__main__':
    # 将属性转换成id
    # attr_change_id2dict()

    # 通过调用 save_dict_as_pickle 函数保存 attr_dict
    # attr_dict = attr_change_id2dict()  # 获取要保存的字典数据
    # file_path = 'data/attr/text_valid_data.pkl'  # 指定保存的文件路径
    # tqdm(save_dict_as_pickle(attr_dict, file_path))  # 调用保存函数
    # path = 'data/dbmg/id2entity.pkl'
    # path = 'data/attr/entity_train_data_features_comet.pkl'
    # tqdm(read_pkl(path))
    # read()
    # extract_text_from_redial()
    # extract_entity_from_redial()
    attr_change_id()

