import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


class dataset(object):
    def __init__(self,filename,opt):
        self.entity2entityId=pkl.load(open('../data/dbmg/entity2entityId.pkl', 'rb'))
        # self.entity2entityId=json.load(open('data/lmkg/lmkg_entity2id.json', 'rb'))
        self.entity_max=len(self.entity2entityId)
        self.id2entity=pkl.load(open('../data/dbmg/id2entity.pkl', 'rb'))
        # self.id2entity=json.load(open('data/lmkg/lmkg_id2entity.json', 'rb'))
        self.subkg=pkl.load(open('../data/dbmg/DBMG_subkg.pkl', 'rb'))
        # 这个里面的key是lmkg_entity2id.json里面的key
        # self.subkg=json.load(open('data/lmkg/lmkg.json' , 'r', encoding='utf-8'))
        self.text_dict=pkl.load(open('../data/introduction/text_dict_no@_0.8_intro_1.pkl', 'rb'))

        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.max_r_length=opt['max_r_length']
        self.max_count=opt['max_count']
        self.entity_num=opt['n_entity']

        with open('../data/review/movie2tokenreview_helpful.pkl', 'rb') as f:
            dict_movieid2review = pkl.load(f)
        self.reviews = dict_movieid2review
        # ---------
        with open('../data/introduction/moveid2introduction_dict.pkl', 'rb') as f:
            dict_moveid2introduction = pkl.load(f)
        self.introduction = dict_moveid2introduction
        # ---------

        ############################################
        with open('../data/attr/attr_dict.pkl', 'rb') as f:
            attribute = pkl.load(f)
        self.attribute = attribute
        ############################################

        f=open(filename,encoding='utf-8')
        self.data=[]
        self.corpus=[]
        total_rev_count=0
        total_intro_count=0
        total_intro_entity=0
        total_entity_attr = 0
        for line in tqdm(f):
            lines=json.loads(line.strip())
            seekerid=lines["initiatorWorkerId"]
            recommenderid=lines["respondentWorkerId"]
            contexts=lines['messages']
            movies=lines['movieMentions']
            altitude=lines['respondentQuestions']
            initial_altitude=lines['initiatorQuestions']
            cases, rev_count, intro_count, intro_entity_len, entity_attr_len = self._context_reformulate(contexts,movies,altitude,initial_altitude,seekerid,recommenderid)
            total_rev_count += rev_count
            total_intro_count += intro_count
            total_intro_count += intro_count
            total_intro_entity += intro_entity_len
            total_entity_attr += entity_attr_len
            self.data.extend(cases)
        print("total_rev_count:",total_rev_count,", total_intro_count:",total_intro_count, ", total_intro_entity:",total_intro_entity, "total_entity_attr:",total_entity_attr)

        self.key2index=json.load(open('../data/word/key2index_3rd.json', encoding='utf-8'))
        self.stopwords=set([word.strip() for word in open('../data/word/stopwords.txt', encoding='utf-8')])
        self.keyword_sets, self.movie_wordset = self.co_occurance_ext(self.data)
        # self.prepare_word2vec()
        self.word2index = json.load(open('../data/word/word2index_redial_intro.json', encoding='utf-8'))

    def prepare_word2vec(self):
        import gensim
        model=gensim.models.word2vec.Word2Vec(self.corpus,vector_size=300,min_count=1)
        model.save('word2vec_redial_intro')
        word2index = {word: i + 4 for i, word in enumerate(model.wv.index_to_key)}

        word2embedding = [[0] * 300] * 4 + [model.wv[word] for word in word2index]+[[0]*300]
        
        word2index['_split_']=len(word2index)+4
        json.dump(word2index, open('../data/word/word2index_redial_intro.json', 'w', encoding='utf-8'), ensure_ascii=False)
        #print(len(word2index)+4)

        import numpy as np
        #print(np.shape(word2embedding))
        np.save('../data/word/word2vec_redial_intro.npy', word2embedding)

        #########################
        mask4key = np.zeros(len(word2index)+4)
        mask4movie = np.zeros(len(word2index)+4)
        for word in word2index:
            idx = word2index[word]
            if word in self.keyword_sets:
                mask4key[idx] = 1
            if word in self.movie_wordset:
                mask4movie[idx] = 1
        print('mask4keyshape:',mask4key.shape, mask4key.sum())
        print('mask4movieshape:',mask4movie.shape, mask4movie.sum())
        np.save('../data/introduction/mask4key20intro.npy', mask4key)
        np.save('../data/introduction/mask4movie20intro.npy', mask4movie)
        print("prepare_word2vec complected!")


    def padding_w2v(self,sentence,max_length,transformer=True,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        reviews_mask=[]
        introduction_mask=[]
        for word in sentence:
            #### vector ####
            if '#' in word or '$' in word:
                vector.append(self.word2index.get(word[1:],unk))
            else:
                vector.append(self.word2index.get(word,unk))
            
            #### concept_mask ####
            concept_mask.append(self.key2index.get(word.lower(),0))
            
            #### dbpedia_mask ####
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    id=self.entity2entityId[entity]
                except:
                    id=self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
            
            #### review_mask ####
            if '#' in word:
                reviews_mask.append(1)#self.word2index.get(word[1:],unk))
            else:
                reviews_mask.append(0)#pad)

            if '$' in word:
                introduction_mask.append(1)
            else:
                introduction_mask.append(0)
                
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)
        reviews_mask.append(0)#pad)
        introduction_mask.append(0)#pad)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:],reviews_mask[:max_length],introduction_mask[:max_length]
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length],reviews_mask[:max_length],introduction_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],\
                   dbpedia_mask+(max_length-len(vector))*[self.entity_max],\
                   reviews_mask+(max_length-len(vector))*[0], \
                   introduction_mask + (max_length - len(vector))*[0]
    
    def padding_context(self,contexts,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        if transformer==False:
            if len(contexts)>self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.max_count-length)*[[pad]*self.max_c_length],vec_lengths+[0]*(self.max_count-length),length
        else:
            contexts_com=[]
            for sen in contexts[-self.max_count:-1]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            vec,v_l,concept_mask,dbpedia_mask,reviews_mask, introduction_mask = self.padding_w2v(contexts_com,self.max_c_length,transformer)
            return vec,v_l,concept_mask,dbpedia_mask,reviews_mask,introduction_mask,0

    def response_delibration(self,response,unk='MASKED_WORD'):
        new_response=[]
        for word in response:
            if word in self.key2index:
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def data_process(self,is_finetune=False):
        data_set = []
        context_before = []
        for line in self.data:
            if is_finetune and line['contexts'] == context_before:
                continue
            else:
                context_before = line['contexts']
            context,c_lengths,concept_mask,dbpedia_mask,reviews_mask,introduction_mask,_=self.padding_context(line['contexts'])
            response,r_length,_,_,_,_=self.padding_w2v(line['response'],self.max_r_length)
            if False:
                mask_response,mask_r_length,_,_,_,_=self.padding_w2v(self.response_delibration(line['response']),self.max_r_length)
            else:
                mask_response, mask_r_length=response,r_length
            assert len(context)==self.max_c_length
            assert len(concept_mask)==self.max_c_length
            assert len(dbpedia_mask)==self.max_c_length

            data_set.append([np.array(context),c_lengths,np.array(response),r_length,np.array(mask_response),mask_r_length,line['entity'],line['entities_altitude'],line['entities_altitude_attr'],
                             line['movie'], line['movie_altitude'], concept_mask,dbpedia_mask,reviews_mask,introduction_mask,line['rec']])
        return data_set

    def co_occurance_ext(self,data):
        stopwords=set([word.strip() for word in open('../data/word/stopwords.txt', encoding='utf-8')])
        keyword_sets=set(self.key2index.keys())-stopwords
        movie_wordset=set()
        for line in data:
            movie_words=[]
            if line['rec']==1:
                for word in line['response']:
                    if '@' in word:
                        try:
                            num=self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line['movie_words']=movie_words
        new_edges=set()
        for line in data:
            if len(line['movie_words'])>0:
                before_set=set()
                after_set=set()
                co_set=set()
                for sen in line['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before'+'\t'+movie+'\t'+word+'\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in line['movie_words']:
                        if word!=movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after'+'\t'+word+'\t'+movie+'\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after'+'\t'+word+'\t'+word_a+'\n')
        f=open('../genrate_conv/co_occurance.txt', 'w', encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset),open('../data/dataset/movie_word.json','w',encoding='utf-8'),ensure_ascii=False)
        json.dump(list(keyword_sets), open('../data/word/key_word.json', 'w', encoding='utf-8'), ensure_ascii=False)
        print('len(new_edges):',len(new_edges))
        print('len(keyword_sets)',len(keyword_sets))
        print('len(movie_wordset)',len(movie_wordset)) # RevCore 2314,626  # MICR 3797,1195 || 去掉@ 1524  # 2573

        return keyword_sets, movie_wordset
    
    def detect_movie(self,sentence, movies):
        token_text = word_tokenize(sentence)
        num=0
        token_text_com=[]
        intro_count=0
        rev_count=0
        entities = []
        info_attr = []
        entity_attr = []
        all_attr = []
        addReview = True
        addIntro = True
        while num<len(token_text):
            if token_text[num]=='@' and num+1<len(token_text):

                # **************************************************
                for k in self.attribute.keys():
                    movie_id = token_text[num + 1]
                    if k == movie_id:
                        movie_data = self.attribute[k]
                        language = movie_data['Language']
                        produce = movie_data['Produced_by']
                        star = movie_data['Starring']
                        direct = movie_data['Directed_by']
                        writer = movie_data['Writer_by']
                        all_attr = language + produce + star + direct + writer
                        break
                info_attr.extend(list(set(all_attr)))
                # **************************************************

                if addReview and token_text[num+1] in self.reviews:
                    token_review = self.reviews[token_text[num+1]]
                    # token_review = ['#'+word_rev for word_rev in token_review]
                    token_text_com.append(token_text[num]+token_text[num+1])
                    token_text_com += token_review
                    rev_count += 1
                    addIntro = False
                    addReview = False
                elif addIntro and token_text[num+1] in self.introduction:
                    token_introduction = self.introduction[token_text[num + 1]]
                    try:
                        for entity in self.text_dict[' '.join(token_introduction)]:
                            try:
                                entities.append(self.entity2entityId[entity])
                                entity_attr.append(info_attr)
                                # **************************************************
                                if entities == []:
                                    pass
                                else:
                                    for entity in entities:
                                        for k in self.attribute.keys():
                                            if k == entity:
                                                entity_attr.append(self.attribute[k])
                                                entity_attr.append(info_attr)
                                # **************************************************

                            except:
                                pass
                    except:
                        pass
                    # token_introduction = ['$'+word_rev for word_rev in token_introduction]
                    token_text_com.append(token_text[num] + token_text[num + 1])
                    # When adding a introduction, add it to the end of the sentence
                    token_text_com += token_introduction
                    intro_count += 1
                    addReview = False
                    addIntro = False
                else:
                    token_text_com.append(token_text[num]+token_text[num+1])
                num+=2
            else:
                token_text_com.append(token_text[num])
                num+=1

        movie_rec = []
        for word in token_text_com[:-30]:
            if word[1:] in movies:
                movie_rec.append(word[1:])
        movie_rec_trans=[]
        for movie in movie_rec:
            entity = self.id2entity[int(movie)]
            try:
                movie_rec_trans.append(self.entity2entityId[entity])
            except:
                pass
        return token_text_com, movie_rec_trans, rev_count, intro_count, list(set(entities)), list(set(info_attr))

    def _context_reformulate(self,context,movies,altitude,ini_altitude,s_id,re_id):
        # ------
        ini_altitude_entityid = dict()
        # Convert ini_altitude to a dictionary with entity id as key, which was originally movie id as key
        if len(ini_altitude) > 0:
            for k, v in ini_altitude.items():
                try:
                    entity = self.id2entity[int(k)]
                    ini_altitude_entityid[self.entity2entityId[entity]] = v
                except:
                    pass
            # ***************************************************
            for k, v in ini_altitude.items():
                try:
                    for k_attr in self.attribute.keys():
                        if k == k_attr:
                            ini_altitude_entityid[self.entity2entityId[entity]] = v
                except:
                    pass
            # ***************************************************
        # ------
        last_id=None
        context_list=[]
        info_attr = []
        rev_count = 0
        intro_count = 0
        intro_entity_len = 0
        entity_attr_len = 0
        for message in context:
            entities=[]
            entity_attr = []
            try:
                for entity in self.text_dict[message['text']]:
                    try:
                        entities.append(self.entity2entityId[entity])
                    except:
                        pass

                # ******************************************
                for entity in self.text_dict[message['text']]:
                    entityId = self.entity2entityId(entity)
                    # ******************************************
                    for k in self.attribute.keys():
                        try:
                            if k == entityId:
                                movie_data = self.attribute[k]
                                language = movie_data['Language']
                                produce = movie_data['Produced_by']
                                star = movie_data['Starring']
                                direct = movie_data['Directed_by']
                                writer = movie_data['Writer_by']
                                all_attr = language + produce + star + direct + writer
                                break
                            entity_attr.extend(list(set(all_attr)))
                        except:
                            pass
                    # ******************************************
                    for k in self.attribute.keys():
                        try:
                            if entityId == k:
                                entity_attr.append(self.attribute[k])
                                entity_attr.append(info_attr)
                        except:
                            pass
                # ******************************************
            except:
                pass

            token_text, movie_rec, temp_rev_count, temp_intro_count, intro_entity, entity_attr = self.detect_movie(message['text'], movies)

            if len(intro_entity) > 0:
                entities.extend(intro_entity)
                intro_entity_len += len(intro_entity)

            # ******************************************
            if len(entity_attr) > 0:
                entity_attr.append(entity_attr)
                entity_attr_len += len(entity_attr)
            # ******************************************

            rev_count += temp_rev_count
            intro_count += temp_intro_count
            final_entity = entities + movie_rec
            # ******************************************
            # final_entity = entities + movie_rec + entity_attr
            # ******************************************
            final_entity = list(set(final_entity))

            if len(context_list)==0:
                # context_dict={'text':token_text, 'entity':final_entity, 'user':message['senderWorkerId'], 'movie':movie_rec}

                # ******************************************'entity_attr':entity_attr,
                context_dict={'text':token_text, 'entity': final_entity, 'entity_attr':entity_attr, 'user':message['senderWorkerId'], 'movie':movie_rec}
                # ******************************************'entity_attr':entity_attr,

                context_list.append(context_dict)
                last_id=message['senderWorkerId']
                continue
            if message['senderWorkerId']==last_id:
                context_list[-1]['text']+=token_text
                context_list[-1]['entity']+=entities+movie_rec
                # ******************************************'entity_attr':entity_attr,
                context_list[-1]['entity_attr'] += entity_attr
                # ******************************************'entity_attr':entity_attr,
                context_list[-1]['movie']+=movie_rec
            else:
                # context_dict = {'text': token_text, 'entity': entities+movie_rec,'user': message['senderWorkerId'], 'movie': movie_rec}

                # ******************************************'entity_attr':entity_attr,
                context_dict = {'text': token_text, 'entity': final_entity, 'entity_attr': entity_attr,
                                'user': message['senderWorkerId'], 'movie': movie_rec}
                # ******************************************'entity_attr':entity_attr,

                context_list.append(context_dict)
                last_id = message['senderWorkerId']

        cases=[]
        contexts=[]
        entities_set=set()
        entities=[]
        entity_attr = []
        for context_dict in context_list:
            self.corpus.append(context_dict['text'])
            # -------------原始
            # 0 means the user does not like the movie, 1 means he likes it, 2 means he does not know
            # entities_altitude = dict()
            # for entityId in entities:
            #     if entityId in ini_altitude_entityid.keys():
            #         entities_altitude[entityId] = ini_altitude_entityid[entityId]['liked']
            #     else:
            #         entities_altitude[entityId] = 2

            # -------------

            # *********************************************************
            # 0 means the user does not like the movie, 1 means he likes it, 2 means he does not know
            entities_altitude = dict()
            entities_altitude_attr = dict()
            for entityId in entities:
                if entityId in ini_altitude_entityid.keys():

                    entities_altitude[entityId] = ini_altitude_entityid[entityId]['liked']
                else:
                    entities_altitude[entityId] = 2

            for entityId in entities:
                for k in self.attribute.keys():
                    if entityId == k:
                        entities_altitude_attr[entityId]= ini_altitude_entityid[entityId]['liked']
                        entity_attr.append(self.attribute[k])
                    else:
                        entities_altitude[entityId] = 2

            # *********************************************************

            # If the current sentence is a recommendation
            if context_dict['user']==re_id and len(contexts)>0:
                response=context_dict['text']
                movie_altitude = -1
                if len(context_dict['movie']) != 0:
                    # Split the recommended movies with the same context, but iterate through the recommended movies each time; i.e. only one movie will be recommended each time
                    for movie in context_dict['movie']:
                        if movie in ini_altitude_entityid.keys():
                            movie_altitude = ini_altitude_entityid[movie]['liked']
                        # cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities),'movie': movie, 'entities_altitude': entities_altitude, 'movie_altitude':movie_altitude,'rec': 1})
                        cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'entity_attr':entity_attr,'movie': movie, 'entities_altitude': entities_altitude,'entities_altitude_attr':entities_altitude_attr, 'movie_altitude':movie_altitude,'rec': 1})
                else:
                    # Even if there are no recommendations, there may still be movie entities in entities
                    # cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': 0,'entities_altitude': entities_altitude,'movie_altitude':movie_altitude,'rec': 0})
                    cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'entity_attr':entity_attr, 'entities_altitude_attr':entities_altitude_attr, 'movie': 0,'entities_altitude': entities_altitude,'movie_altitude':movie_altitude,'rec': 0})
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
            else:
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)

        return cases,rev_count,intro_count, intro_entity_len, entity_attr_len


class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1
        print('data with review')

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context, c_lengths, response, r_length, mask_response, mask_r_length, entity,entities_altitude, entities_altitude_attr, movie, movie_altitude, concept_mask, dbpedia_mask, reviews_mask, introduction_mask, rec = self.data[index]
        entity_vec = np.zeros(self.entity_num)
        entity_vector=np.zeros(60,dtype=np.int)
        point=0
        for en in entity:
            entity_vec[en]=1
            entity_vector[point]=en
            point+=1

        # Set the corresponding position of the preferred entity to 1, dislike to -1, and uncertainty to 2
        entities_altitude_vec = np.zeros(self.entity_num)
        for k, v in entities_altitude.items():
            if v == 0:  # 不喜欢
                entities_altitude_vec[k] = -1
            else:
                entities_altitude_vec[k] = v

        entities_altitude_attr_vec = np.zeros(self.entity_num)
        for k, v in entities_altitude_attr.items():
            if v == 0:  # 不喜欢
                entities_altitude_attr_vec[k] = -1
            else:
                entities_altitude_attr_vec[k] = v

        concept_vec=np.zeros(self.concept_num)
        for con in concept_mask:
            if con!=0:
                concept_vec[con]=1

        db_vec=np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db!=0:
                db_vec[db]=1

        return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, entities_altitude_vec, entities_altitude_attr_vec, movie, movie_altitude,np.array(concept_mask), np.array(dbpedia_mask), np.array(reviews_mask), np.array(introduction_mask), concept_vec, db_vec, rec

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    ds=dataset('../data/dataset/train_data.jsonl')
    print()
