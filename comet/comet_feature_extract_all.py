import pickle, numpy as np

from comet.csk_feature_extract import CSKFeatureExtractor

extractor = CSKFeatureExtractor()

for dataset in ['entity_train_data', 'entity_test_data', 'entity_valid_data']:
    print('Extracting features in', dataset)
    path = '../data/attr'
    sentences = pickle.load(open(path + '/' + dataset + '.pkl', 'rb'))
    features = extractor.extract(sentences)
    # 打开文件并将其分配给变量
    file = open(path + '/' + dataset + '_features_comet.pkl', 'wb')
    # 将特征序列化到文件
    pickle.dump(features, file)
print('Done!')