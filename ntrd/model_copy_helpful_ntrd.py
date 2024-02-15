# coding:utf-8
import os
import pickle
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
from models.bert_comet import bert_redial
from models.bert_comet import fusion_comet_bert
from models.transformer import _build_decoder_selection
from models.utils import grad_mul_const

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

EDGE_TYPES = [58, 172]


def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings


def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[
                    0] != 185:  # and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)


def concept_edge_list4GCN():
    node2index = json.load(open('../data/word/key2index_3rd.json', encoding='utf-8'))
    f = open('../data/word/conceptnet_edges2nd.txt', encoding='utf-8')
    edges = set()
    stopwords = set([word.strip() for word in open('../data/word/stopwords.txt', encoding='utf-8')])
    for line in f:
        lines = line.strip().split('\t')
        entity0 = node2index[lines[1].split('/')[0]]
        entity1 = node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0, entity1))
        edges.add((entity1, entity0))
    edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).cuda()


class CrossModel(nn.Module):
    def __init__(self, opt, dictionary, is_finetune=False, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']

        # ---------------
        self.genre_len = opt['n_genre']
        with open('../data/genre/entityId2genre_dict.pkl', 'rb') as f:
            dict_entityId2genre = pkl.load(f)
        self.genre = dict_entityId2genre
        self.is_finetune = is_finetune
        with open('../data/genre/genreId2entityId.pkl', 'rb') as f:
            genreId2entityId = pkl.load(f)
        self.genreId2entityId = genreId2entityId
        # -----------------------------

        # *************************************************************
        self.attr_len = opt['n_attr']
        with open('../data/attr/attr_dict.pkl', 'rb') as f:
            dict_attr = pkl.load(f)
        self.attr = dict_attr
        # 情感COMET使用
        attr_list, common, unique = bert_comet.get_common()
        self.attribute_embedding = AttributeSelfAttentionEmbedding(input_size=opt['dim'], output_size=opt['dim'])
        self.attr_list = attr_list
        if common is not None and unique is not None:
            common = torch.tensor(common)
            unique = torch.tensor(unique)
            self.common = self.attribute_embedding(common)
            self.unique = self.attribute_embedding(unique)

        # 情感分析
        self.bert_redial = bert_redial()
        for k, v in self.bert_redial.items():
            positive_weight = v['positive_weight']
            negative_weight = v['negative_weight']
            neutral_weight = v['neutral_weight']
            positive_label = v['positive_label']
            break
        # 现在只需要把这个权重赋给对应的属性就可以了
        self.bert_encoder = {'positive_weight': positive_weight, 'negative_weight': negative_weight,
                             'neutral_weight': neutral_weight, 'positive_label': positive_label}
        self.max_attr_weight = max(self.bert_encoder['positive_weight'], self.bert_encoder['negative_weight'], self.bert_encoder['neutral_weight'])
        self.common = self.common * self.max_attr_weight
        self.unique = self.unique * (self.bert_encoder['negative_weight'] + self.bert_encoder['neutral_weight']) / 2

        self.comet = pickle.load(open('../data/attr/entity_train_data_features_comet.pkl', 'rb'))
        # 这个是key value的形式
        self.fusion_comet_bert = fusion_comet_bert()
        self.beam = opt['beam']
        self.movieID2selection_label = pkl.load(open('../data/attr/movieID2selection_label.pkl', 'rb'))
        # *************************************************************

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings = _create_entity_embeddings(
            opt['n_concept'] + 1, opt['dim'], 0)
        self.concept_padding = 0

        # self.kg = json.load(
        #     open("data/lmkg/lmkg.json", "r")
        # )
        self.kg = pkl.load(open('../data/dbmg/DBMG_subkg.pkl', 'rb'))

        self.entity2entityId = pkl.load(open('../data/dbmg/entity2entityId.pkl', 'rb'))
        self.id2entity = pkl.load(open('../data/dbmg/id2entity.pkl', 'rb'))

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.selection_cross_attn_decoder = _build_decoder_selection(
            opt, len(self.movieID2selection_label), self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.db_attn_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_attn_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])
        ##################
        self.self_attn_rv = SelfAttentionLayer_batch(opt['embedding_size'], opt['embedding_size'])
        self.self_attn_intro = SelfAttentionLayer_batch(opt['embedding_size'], opt['embedding_size'])
        ##################

        # *************************************************************
        self.self_attn_attr = SelfAttentionLayer_batch(opt['embedding_size'], opt['embedding_size'])
        self.self_attr_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_mm_attr = SelfAttentionLayer(opt['dim'], opt['dim'])
        # *************************************************************

        self.user_norm = nn.Linear(opt['dim'] * 2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        # -----------
        self.user_norm_genre = nn.Linear(opt['dim'] * 3, opt['dim'])
        self.gate_norm_genre = nn.Linear(opt['dim'], 3)
        # -----------

        # *************************************************************
        self.user_norm_attr = nn.Linear(opt['dim'] * 3, opt['dim'])
        self.gate_norm_attr = nn.Linear(opt['dim'], 3)
        # *************************************************************

        self.copy_norm = nn.Linear(opt['embedding_size'] * 2 + opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)
        ##################
        self.vocab_size = len(dictionary) + 4
        self.copy_norm_rev = nn.Linear(opt['embedding_size'] + opt['embedding_size'], opt['embedding_size'])
        self.representation_bias_rev = nn.Linear(opt['embedding_size'], len(dictionary) + 4)
        self.copy_norm_intro = nn.Linear(opt['embedding_size'] + opt['embedding_size'], opt['embedding_size'])
        self.representation_bias_intro = nn.Linear(opt['embedding_size'], len(dictionary) + 4)
        ##################

        self.info_con_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_db_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept'] + 1)
        self.info_con_loss = nn.MSELoss(size_average=False, reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False, reduce=False)
        # -------
        self.info_genre_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_genre = nn.Linear(opt['dim'], opt['n_entity'])
        # -------

        # *************************************************************
        self.info_attr_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_attr = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_seed_attr_norm = nn.Linear(opt['dim'], opt['dim'])
        self.linear_dim2_emb = nn.Linear(opt['embedding_size'], opt['dim'])
        # *************************************************************

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)

        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size = opt['embedding_size']
        self.dim = opt['dim']

        edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2)
        edge_list = list(set(edge_list))
        # print(len(edge_list), self.n_relation)
        self.dbpedia_edge_sets = torch.LongTensor(edge_list).cuda()
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN = RGCNConv(opt['n_entity'], self.dim, self.n_relation, num_bases=opt['num_bases'])
        # self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.n_con_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets = concept_edge_list4GCN()
        self.concept_GCN = GCNConv(self.dim, self.dim)

        # self.concept_GCN4gen=GCNConv(self.dim, opt['embedding_size'])

        w2i = json.load(open('../data/word/word2index_redial_intro.json', encoding='utf-8'))
        self.i2w = {w2i[word]: word for word in w2i}

        self.mask4key = torch.Tensor(np.load('../data/introduction/mask4key20intro.npy')).cuda()
        self.mask4movie = torch.Tensor(np.load('../data/introduction/mask4movie20intro.npy')).cuda()
        self.mask4 = self.mask4key + self.mask4movie
        if is_finetune:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.concept_embeddings.parameters(),
                      self.self_attn.parameters(), self.self_attn_db.parameters(), self.user_norm.parameters(),
                      self.gate_norm.parameters(),
                      # 原始内容
                      # self.output_en.parameters(),self.gate_norm_genre.parameters(), self.user_norm_genre.parameters()]
                      # *************************************************************
                      self.output_en.parameters(), self.gate_norm_genre.parameters(), self.user_norm_genre.parameters(),
                      self.gate_norm_attr.parameters(), self.gate_norm_attr.parameters()]
            # *************************************************************

            for param in params:
                for pa in param:
                    pa.requires_grad = False

    def vector2sentence(self,batch_sen):
        sentences=[]
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    sentence.append(self.index2word[word])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def decode_greedy(self, encoder_states, xs_rev, xs_intro, encoder_states_kg, encoder_states_db, attention_kg,
                      attention_db, attention_rv, attention_intro, bsz, maxlen):
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        latents = []
        for i in range(maxlen):
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            scores = scores[:, -1:, :]
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            db_attn_norm = self.db_attn_norm(attention_db)

            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))
            latents.append(scores)  # WAY2:self attn matching model
            copy_latent_rev = self.copy_norm_rev(torch.cat([attention_rv.unsqueeze(1), scores], -1))
            copy_latent_intro = self.copy_norm_intro(torch.cat([attention_intro.unsqueeze(1), scores], -1))

            con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(0)
            voc_logits = F.linear(scores, self.embeddings.weight)

            #############################
            mask4_rev = torch.zeros(bsz, self.vocab_size)
            index_list_mask_rev = xs_rev.tolist()
            mask4_rev = mask4_rev.scatter_(1, torch.LongTensor(index_list_mask_rev),
                                           torch.ones(bsz, self.vocab_size)).cuda()
            con_logits_rev = self.representation_bias_rev(copy_latent_rev) * mask4_rev.unsqueeze(1)

            mask4_intro = torch.zeros(bsz, self.vocab_size)
            index_list_mask_intro = xs_intro.tolist()
            mask4_intro = mask4_intro.scatter_(1, torch.LongTensor(index_list_mask_intro),
                                               torch.ones(bsz, self.vocab_size)).cuda()
            con_logits_intro = self.representation_bias_intro(copy_latent_intro) * mask4_intro.unsqueeze(1)
            #############################

            sum_logits = voc_logits + con_logits + con_logits_rev + con_logits_intro
            _, preds = sum_logits.max(dim=-1)

            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        latents = torch.cat(latents, 1)
        return logits, xs, latents

    def decode_beam_search_with_kg(self, token_encoding, encoder_states_kg, encoder_states_db, attention_kg,
                                   attention_db, maxlen=None, beam=4):
        entity_reps, entity_mask = encoder_states_db
        word_reps, word_mask = encoder_states_kg
        entity_emb_attn = attention_db
        word_emb_attn = attention_kg
        batch_size = token_encoding[0].shape[0]

        inputs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        incr_state = None

        sequences = [[[list(), list(), 1.0]]] * batch_size
        all_latents = []
        # for i in range(self.response_truncate):
        for i in range(maxlen):
            if i == 1:
                token_encoding = (token_encoding[0].repeat(beam, 1, 1),
                                  token_encoding[1].repeat(beam, 1, 1))
                entity_reps = entity_reps.repeat(beam, 1, 1)
                entity_emb_attn = entity_emb_attn.repeat(beam, 1)
                entity_mask = entity_mask.repeat(beam, 1)
                word_reps = word_reps.repeat(beam, 1, 1)
                word_emb_attn = word_emb_attn.repeat(beam, 1)
                word_mask = word_mask.repeat(beam, 1)

                encoder_states_kg = word_reps, word_mask
                encoder_states_db = entity_reps, entity_mask

            # at beginning there is 1 candidate, when i!=0 there are 4 candidates
            if i != 0:
                inputs = []
                for d in range(len(sequences[0])):
                    for j in range(batch_size):
                        text = sequences[j][d][0]
                        inputs.append(text)
                inputs = torch.stack(inputs).reshape(beam, batch_size, -1)  # (beam, batch_size, _)

            with torch.no_grad():

                dialog_latent, incr_state = self.decoder(inputs.reshape(len(sequences[0]) * batch_size, -1),
                                                         token_encoding, encoder_states_kg, encoder_states_db,
                                                         incr_state)
                # dialog_latent, incr_state = self.conv_decoder(
                #     inputs.reshape(len(sequences[0]) * batch_size, -1),
                #     token_encoding, word_reps, word_mask,
                #     entity_reps, entity_mask, incr_state
                # )
                dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)

                concept_latent = self.kg_attn_norm(word_emb_attn).unsqueeze(1)
                db_latent = self.db_attn_norm(entity_emb_attn).unsqueeze(1)

                # print('concept_latent shape', concept_latent.shape)
                # print('db_latent shape', db_latent.shape)
                # print('dialog_latent shape', dialog_latent.shape)

                copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

                all_latents.append(copy_latent)

                # copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
                copy_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(0)
                gen_logits = F.linear(dialog_latent, self.embeddings.weight)
                sum_logits = copy_logits + gen_logits

            logits = sum_logits.reshape(len(sequences[0]), batch_size, 1, -1)
            # turn into probabilities,in case of negative numbers
            probs, preds = torch.nn.functional.softmax(logits).topk(beam, dim=-1)

            # (candeidate, bs, 1 , beam) during first loop, candidate=1, otherwise candidate=beam

            for j in range(batch_size):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        prob = sequences[j][n][2]
                        logit = sequences[j][n][1]
                        if logit == []:
                            logit_tmp = logits[n][j][0].unsqueeze(0)
                        else:
                            logit_tmp = torch.cat((logit, logits[n][j][0].unsqueeze(0)), dim=0)
                        seq_tmp = torch.cat((inputs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                        candidate = [seq_tmp, logit_tmp, prob * probs[n][j][0][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
                sequences[j] = ordered[:beam]

            # check if everyone has generated an end token
            all_finished = ((inputs == self.END_IDX).sum(dim=1) > 0).sum().item() == batch_size
            if all_finished:
                break

        # original solution
        # logits = torch.stack([seq[0][1] for seq in sequences])
        # inputs = torch.stack([seq[0][0] for seq in sequences])

        out_logits = []
        out_preds = []
        for beam_num in range(beam):
            cur_out_logits = torch.stack([seq[beam_num][1] for seq in sequences])
            curout_preds = torch.stack([seq[beam_num][0] for seq in sequences])
            out_logits.append(cur_out_logits)
            out_preds.append(curout_preds)

        logits = torch.cat([x for x in out_logits], dim=0)
        inputs = torch.cat([x for x in out_preds], dim=0)
        all_latents = torch.cat(all_latents, 1)

        return logits, inputs, all_latents

    def decode_forced(self, encoder_states, xs_rev, xs_intro, encoder_states_kg, encoder_states_db, attention_kg,
                      attention_db, attention_rv, attention_intro, ys):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db)

        kg_attention_latent = self.kg_attn_norm(attention_kg)
        db_attention_latent = self.db_attn_norm(attention_db)
        copy_latent = self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
                                                db_attention_latent.unsqueeze(1).repeat(1, seqlen, 1), latent], -1))

        copy_latent_rev = self.copy_norm_rev(torch.cat([attention_rv.unsqueeze(1).repeat(1, seqlen, 1), latent], -1))
        copy_latent_intro = self.copy_norm_intro(
            torch.cat([attention_intro.unsqueeze(1).repeat(1, seqlen, 1), latent], -1))

        logits = F.linear(latent, self.embeddings.weight)
        con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(0)
        #############################
        mask4_rev = torch.zeros(bsz, self.vocab_size)
        index_list_mask_rev = xs_rev.tolist()
        mask4_rev = mask4_rev.scatter_(1, torch.LongTensor(index_list_mask_rev),
                                       torch.ones(bsz, self.vocab_size)).cuda()
        con_logits_rev = self.representation_bias_rev(copy_latent_rev) * mask4_rev.unsqueeze(1)

        mask4_intro = torch.zeros(bsz, self.vocab_size)
        index_list_mask_intro = xs_intro.tolist()
        mask4_intro = mask4_intro.scatter_(1, torch.LongTensor(index_list_mask_intro),
                                           torch.ones(bsz, self.vocab_size)).cuda()
        con_logits_intro = self.representation_bias_intro(copy_latent_intro) * mask4_intro.unsqueeze(1)
        #############################
        sum_logits = logits + con_logits + con_logits_rev + con_logits_intro
        _, preds = sum_logits.max(dim=2)
        return logits, preds, latent

    def infomax_loss(self, con_nodes_features, db_nodes_features, con_user_emb, db_user_emb, genre_user_emb,
                     attr_user_emb, con_label, db_label, mask):
        # batch*dim
        # node_count*dim
        con_emb = self.info_con_norm(con_user_emb)
        db_emb = self.info_db_norm(db_user_emb)
        genre_emb = self.info_genre_norm(genre_user_emb)

        # *************************************************************
        # 对属性信息进行损失函数的维度变换
        attr_emb = self.info_attr_norm(attr_user_emb)
        # *************************************************************

        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)
        genre_scores = F.linear(genre_emb, db_nodes_features, self.info_output_genre.bias)

        # *************************************************************
        attr_scores = F.linear(attr_emb, db_nodes_features, self.info_output_attr.bias)
        # *************************************************************

        info_db_loss = torch.sum(self.info_db_loss(db_scores, db_label.cuda().float()), dim=-1) * mask.cuda()
        info_con_loss = torch.sum(self.info_con_loss(con_scores, con_label.cuda().float()), dim=-1) * mask.cuda()
        info_genre_loss = torch.sum(self.info_con_loss(genre_scores, db_label.cuda().float()), dim=-1) * mask.cuda()

        # *************************************************************
        info_attr_loss = torch.sum(self.info_con_loss(attr_scores, db_label.cuda().float()), dim=-1) * mask.cuda()
        # *************************************************************

        # return torch.mean(info_db_loss), torch.mean(info_con_loss), torch.mean(info_genre_loss)
        # *************************************************************
        return torch.mean(info_db_loss), torch.mean(info_con_loss), torch.mean(info_genre_loss), torch.mean(
            info_attr_loss)
        # *************************************************************

    def forward(self, xs, ys, mask_ys, concept_mask, db_mask, reviews_mask, introduction_mask, seed_sets,
                entities_sets_altitude, entities_sets_altitude_attr, labels, movie_sets_altitude, con_label, db_label,
                entity_vector, rec, movies_gth=None, movie_nums=None, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        if test == False:
            self.longest_label = max(self.longest_label, ys.size(1))

        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)

        xs_rev = xs.mul(reviews_mask.cuda())
        reviews_states, _ = self.encoder(xs_rev)

        xs_intro = xs.mul(introduction_mask.cuda())
        introduction_states, _ = self.encoder(xs_intro)

        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        con_nodes_features = self.concept_GCN(self.concept_embeddings.weight, self.concept_edge_sets)

        user_representation_list = []
        db_con_mask = []

        genre_representation_list = []
        for i, seed_set in enumerate(seed_sets):
            if int(movie_sets_altitude[i]) == 1 and int(labels[i]) in self.genre.keys():
                genre_representation_list.append(self.self_attn_db(db_nodes_features[
                                                                       [self.genreId2entityId[i.item()] for i in
                                                                        torch.tensor(self.genre[int(labels[i])])]]))
            else:
                genre_representation_list.append(torch.zeros(self.dim).cuda())

        # *********************************************************
        """
        根据实体的态度检索属性信息，如果对应的实体找到了对应的属性，那么把他们利用Self-Attention进行编码
        再添加到列表中去，如果没有检索到，那么全部用零维度进行初始化， 列表中添加的全都是零
        """
        attr_representation_list = []
        for i, seed_set in enumerate(seed_sets):
            if int(movie_sets_altitude[i]) == 1 and int(labels[i]) in self.attr.keys():
                attr_representation_list.append(self.self_attn_db(db_nodes_features[
                                                                      [self.attr[i.item()] for i in
                                                                       torch.tensor(self.attr[int(labels[i])])]]))
            else:
                attr_representation_list.append(torch.zeros(self.dim).cuda())
            # *********************************************************

        # //////////////////////////////////////////////////////////////
        """
        根据实体的态度检索属性信息，如果对应的实体找到了对应的属性，那么把他们利用Self-Attention进行编码
        再添加到列表中去，如果没有检索到，那么全部用零维度进行初始化， 列表中添加的全都是零
        """
        attr_fusion_representation_list2 = []
        for i, seed_set in enumerate(seed_sets):
            if int(movie_sets_altitude[i]) == 1 and int(labels[i]) in self.fusion_comet_bert.keys():
                attr_fusion_representation_list2.append(self.self_attn_db(db_nodes_features[
                                                                      [self.fusion_comet_bert[i.item()] for i in
                                                                       torch.tensor(
                                                                           self.fusion_comet_bert[int(labels[i])])]]))

            else:
                attr_fusion_representation_list2.append(torch.zeros(self.dim).cuda())
            # //////////////////////////////////////////////////////////////

            # #######################################################原始内容
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb = torch.stack(user_representation_list)
        db_con_mask = torch.stack(db_con_mask)
        db_movie_genre_emb = torch.stack(genre_representation_list)
        # #######################################################原始内容

        # new add*********************************************************
        """
        在执行的时候，它们的维度在第361步的时候, con_user_emb的维度是39*128; db_user_emb的维度是39*128; db_movie_attr_emb的维度是39*128;而361步之前都是128*128;
        这个是拿着主次属性进行融合，将得到的结果与DBpedia检索的属性再次融合，相当于融合了两次。
        得到的结果里面有着属性信息和实体信息。
        """
        attr_common_representation_list = []
        attr_fusion_representation_list = []
        if self.common == [] or self.unique == []:
            attr_common_representation_list.append(torch.zeros(self.dim).cuda())
            db_con_mask.append(torch.zeros([1]))
        else:
            common_representation = self.self_attn_db(self.common.cuda())  # 128,
            unique_representation = self.self_attn_db(self.unique.cuda())  # 128,

        attr_fusion1 = torch.mm(common_representation.unsqueeze(0).T.cuda(),
                                unique_representation.unsqueeze(0).cuda())  # 128*128

        attr_fusion2 = torch.matmul(db_user_emb, attr_fusion1)  # 39*128

        # attribute embedding
        attr_fusion_representation_list.append(attr_fusion2.cuda())  # 128*128

        # //////////////////////////////////////////////////////////////
        db_movie_attr_emb2 = torch.stack(attr_fusion_representation_list2)
        db_movie_attr_emb3 = torch.stack(attr_representation_list)
        # attr_fusion_representation_list = attr_fusion_representation_list + attr_fusion_representation_list2.repeat(128, 1)
        # //////////////////////////////////////////////////////////////

        db_movie_attr_emb = torch.stack(attr_fusion_representation_list)

        # //////////////////////////////////////////////////////////////
        db_movie_attr_emb = db_movie_attr_emb * db_movie_attr_emb2 * db_movie_attr_emb3
        # //////////////////////////////////////////////////////////////
        # new add*********************************************************

        graph_con_emb = con_nodes_features[concept_mask.to(torch.int64)]

        # con_emb_mask:填充
        con_emb_mask = concept_mask == self.concept_padding

        con_user_emb = graph_con_emb

        # con_user_emb通过了self.self_attn维度变换
        con_user_emb, attention = self.self_attn(con_user_emb, con_emb_mask.cuda())

        # *************************************************************正常跑
        attr_emb_count = 0
        user_emb_original = self.user_norm(torch.cat([con_user_emb, db_user_emb], dim=-1))
        uc_gate_original = F.sigmoid(self.gate_norm(user_emb_original))
        user_emb = uc_gate_original * db_user_emb + (1 - uc_gate_original) * con_user_emb
        db_movie_attr_emb = db_movie_attr_emb.squeeze(0)
        # print("db_movie_attr_emb: ", db_movie_attr_emb.shape)
        user_emb_attr = self.user_norm_attr(torch.cat([con_user_emb, db_user_emb, db_movie_attr_emb], dim=-1))
        uc_gate_attr = F.softmax(self.gate_norm_attr(user_emb_attr))
        for i in range(user_emb.shape[0]):
            if sum(db_movie_attr_emb[i]) == 0:
                continue
            else:
                user_emb[i] = uc_gate_attr[i, 0] * db_user_emb[i] + uc_gate_attr[i, 1] * con_user_emb[i] + \
                              uc_gate_attr[i, 2] * db_movie_attr_emb[i]
                attr_emb_count += 1
        # *************************************************************

        # ----------
        genre_emb_count = 0
        user_emb_original = self.user_norm(torch.cat([con_user_emb, db_user_emb], dim=-1))
        uc_gate_original = F.sigmoid(self.gate_norm(user_emb_original))
        user_emb = uc_gate_original * db_user_emb + (1 - uc_gate_original) * con_user_emb

        user_emb_genre = self.user_norm_genre(torch.cat([con_user_emb, db_user_emb, db_movie_genre_emb], dim=-1))
        uc_gate_genre = F.softmax(self.gate_norm_genre(user_emb_genre))
        for i in range(user_emb.shape[0]):
            if sum(db_movie_genre_emb[i]) == 0:
                continue
            else:
                user_emb[i] = uc_gate_genre[i, 0] * db_user_emb[i] + uc_gate_genre[i, 1] * con_user_emb[i] + \
                              uc_gate_genre[i, 2] * db_movie_genre_emb[i]
                genre_emb_count += 1
        # ----------

        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias)

        mask_loss = 0

        info_db_loss, info_con_loss, info_genre_loss, info_attr_loss = self.infomax_loss(con_nodes_features,
                                                                                         db_nodes_features,
                                                                                         con_user_emb, db_user_emb,
                                                                                         db_movie_genre_emb,
                                                                                         db_movie_attr_emb, con_label,
                                                                                         db_label, db_con_mask)

        rec_loss = self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda())
        rec_loss = torch.sum(rec_loss * rec.float().cuda())

        self.user_rep = user_emb
        # print("con_user_emb.shape:  ", con_user_emb.shape)
        # print("db_user_emb.shape:  ", db_user_emb.shape)
        # print("db_movie_attr_emb.shape:  ", db_movie_attr_emb.shape)

        # generation---------------------------------------------------------------------------------------------------
        if self.is_finetune:
            encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)
            rev_atten_emb, attention_rev = self.self_attn_rv(reviews_states, reviews_mask.cuda())
            intro_atten_emb, attention_intro = self.self_attn_intro(introduction_states, introduction_mask.cuda())

            con_nodes_features4gen = con_nodes_features  # self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
            con_emb4gen = con_nodes_features4gen[concept_mask.to(torch.int64)]
            con_mask4gen = concept_mask != self.concept_padding
            # kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
            kg_encoding = (self.kg_norm(con_emb4gen), con_mask4gen.cuda())

            db_emb4gen = db_nodes_features[entity_vector.to(torch.int64)]  # batch*50*dim
            db_mask4gen = entity_vector != 0
            # db_encoding = self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
            db_encoding = (self.db_norm(db_emb4gen), db_mask4gen.cuda())

            if test == False:
                movies_gth = movies_gth * (movies_gth != -1)
                # 跑对话不能加这一句
                # assert torch.sum(movies_gth != 0, dim=(0, 1)) == torch.sum((mask_ys == 6), dim=(0, 1))

                scores, preds, latent = self.decode_forced(encoder_states, xs_rev, xs_intro,
                kg_encoding, db_encoding, con_user_emb, db_user_emb, rev_atten_emb, intro_atten_emb,
                mask_ys)
                # print('shape of scores,preds, mask_ys, latent', scores.shape,preds.shape,mask_ys.shape,latent.shape)
                gen_loss = torch.mean(self.compute_loss(scores, mask_ys))

                # -------------------------------- stage2 movie selection loss-------------- by Jokie
                masked_for_selection_token = (mask_ys == 6)

                # -----------------------------------------entity movie gth
                latent = grad_mul_const(latent, 0.0)  # don't backpropagate selection loss to stage 1 decoder
                new_encoder_output, new_encoder_mask = encoder_states
                new_encoder_output = grad_mul_const(new_encoder_output,
                                                    0.0)  # don't backpropagate selection loss to stage 1 decoder
                new_encoder_mask = grad_mul_const(new_encoder_mask,
                                                  0.0)  # don't backpropagate selection loss to stage 1 decoder
                new_encoder_states = new_encoder_output, new_encoder_mask

                matching_tensor, _ = self.selection_cross_attn_decoder(latent, new_encoder_states, db_encoding)
                matching_logits_ = self.linear_dim2_emb(matching_tensor)  # 300 to 128
                matching_logits_ = F.linear(matching_logits_, db_nodes_features, self.output_en.bias)
                # print('matching_logits_ shape,', matching_logits_.shape)
                matching_logits = torch.masked_select(matching_logits_,
                                                      masked_for_selection_token.unsqueeze(-1).expand_as(
                                                          matching_logits_)).view(-1, matching_logits_.shape[-1])

                # W1: greedy
                _, matching_pred = matching_logits.max(dim=-1)   # 87 # [bsz * dynamic_movie_nums]
                # W2: sample
                # matching_pred = torch.multinomial(F.softmax(matching_logits, dim=-1), num_samples=1, replacement=True)
                # 确保 matching_logits 和 movies_gth 的批次大小相同
                # assert matching_logits.shape[0] == movies_gth.shape[
                #     0], "Batch size mismatch between matching_logits and movies_gth"
                movies_gth = torch.masked_select(movies_gth, (movies_gth != 0))  # 50
                """
                num_samples = min(50, matching_logits.shape[0])  # 保证采样的样本数不超过 matching_logits 的样本数

                # 随机采样并确保批次大小相同
                indices = torch.randperm(matching_logits.shape[0])[:num_samples]
                matching_logits_selected = matching_logits[indices]
                
                # 对 movies_gth 进行相应的批次填充或截断
                movies_gth_selected = movies_gth[indices]
                
                # 计算 selection_loss
                selection_loss = torch.mean(self.compute_loss(matching_logits_selected, movies_gth_selected))

                """
                # 原始有错误
                # selection_loss = torch.mean(self.compute_loss(matching_logits, movies_gth))  # movies_gth.squeeze(0):[bsz * dynamic_movie_nums]
                # ValueError: Expected input batch_size (87) to match target batch_size (50).

                # 修改---------------------------------------------------
                num_samples = min(matching_logits.shape[0], matching_logits.shape[0])   # 87 # 保证采样的样本数不超过 matching_logits 的样本数
                indices = torch.randperm(matching_logits.shape[0])[:movies_gth.shape[0]]  # 87
                matching_logits_selected = matching_logits[indices][:movies_gth.shape[0]]  # 87*64368

                # s = nn.Linear(movies_gth.shape[0], matching_logits_selected.shape[0])
                # movies_gth = s(movies_gth)

                # if matching_logits_selected.shape[0] != movies_gth.shape[0]:
                #     raise ValueError("Batch sizes of matching_logits_selected and movies_gth do not match.")
                # selection_loss = torch.mean(self.compute_loss(matching_logits_selected, movies_gth))  # 87*64368    50
                selection_loss = torch.mean(self.compute_loss(scores, mask_ys))
                # 这个正常跑
                # selection_loss = torch.mean(self.compute_loss(scores, mask_ys))
                # use teacher forcing
                # scores, preds = self.decode_forced(
                #     encoder_states, xs_rev, xs_intro,
                #     kg_encoding, db_encoding, con_user_emb, db_user_emb, rev_atten_emb, intro_atten_emb,
                #     mask_ys
                # )
                # gen_loss = torch.mean(self.compute_loss(scores, mask_ys))
                # -------------------------------- stage2 movie selection loss-------------- by Jokie
                masked_for_selection_token = (mask_ys == 6)
            else:
                scores, preds,latent = self.decode_greedy(
                    encoder_states, xs_rev, xs_intro,
                    kg_encoding, db_encoding, con_user_emb, db_user_emb, rev_atten_emb, intro_atten_emb,
                    bsz, maxlen or self.longest_label
                )
                # gen_loss = None
                # #pred here is soft template prediction
                # # --------------post process the prediction to full sentence
                # #-------------------------------- stage2 movie selection loss-------------- by Jokie
                preds_for_selection = preds[:, 1:]  # skip the start_ind
                masked_for_selection_token = (preds_for_selection == 6)

                # -----------------------------------------entity movie gth
                matching_tensor, _ = self.selection_cross_attn_decoder(latent, encoder_states, db_encoding)
                matching_logits_ = self.linear_dim2_emb(matching_tensor)  # 300 to 128
                matching_logits_ = F.linear(matching_logits_, db_nodes_features, self.output_en.bias)
                # print('matching_logits_ shape,', matching_logits_.shape)
                matching_logits = torch.masked_select(matching_logits_,
                                                      masked_for_selection_token.unsqueeze(-1).expand_as(
                                                          matching_logits_)).view(-1, matching_logits_.shape[-1])

                if matching_logits.shape[0] is not 0:
                    # W1: greedy
                    _, matching_pred = matching_logits.max(dim=-1)  # [bsz * dynamic_movie_nums]
                    # W2: sample
                    # matching_pred = torch.multinomial(F.softmax(matching_logits,dim=-1), num_samples=1, replacement=True)
                else:
                    matching_pred = None
                # print('matching_pred', matching_pred.shape)
                # ---------------------------------------------Greedy decode(end)-------------------------------------------

                gen_loss = None
                selection_loss = None
            return scores, preds, entity_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss, info_genre_loss, genre_emb_count, info_attr_loss, attr_emb_count, selection_loss, matching_pred, matching_logits_
        else:
            # return None, None, entity_scores, rec_loss, None, None, info_db_loss, info_con_loss, None, None
            return None, None, entity_scores, rec_loss, None, mask_loss, info_db_loss, info_con_loss, info_genre_loss, genre_emb_count, info_attr_loss, attr_emb_count, None, None, None

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.cuda(), score_view.cuda())
        return loss

    # def save_model(self):
    #     torch.save(self.state_dict(), 'saved_model/net_parameter1.pkl')
    def save_model(self,model_name='../saved_model/net_parameter1.pkl'):
        torch.save(self.state_dict(), model_name)

    def load_model(self,model_name='../saved_model/net_parameter1.pkl'):
        # self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))
        self.load_state_dict(torch.load(model_name), strict= False)
    # def load_model(self):
    #     pretrained = torch.load('saved_model/net_parameter1.pkl')
    #     network_dict = self.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained.items()}
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in network_dict}
    #     network_dict.update(pretrained_dict)
    #     self.load_state_dict(network_dict)
        # self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsqueeze(dim=1)
        output += up_bias
        return output
