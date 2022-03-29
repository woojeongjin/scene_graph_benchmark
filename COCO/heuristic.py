import numpy as np
import spacy
import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import pandas
import gensim.downloader as api
import inflect
p = inflect.engine()

class heuristic:
    def __init__(self, objects_vocab=None):
        self.nlp = spacy.load('en_core_web_sm')
        self.gensim = api.load("word2vec-google-news-300")
        self.wordnet_lemma = WordNetLemmatizer()
        self.objects_vocab = objects_vocab
        self.concept_dictionary = {}
        self.concept_dict_rel = {}
        self.nouns = None
        self.set_concept_dictionary()
        # self.set_hypernym_edges()

    # def set_hypernym_edges(self):
    #     edges = set()
    #     for synset in tqdm(wordnet.all_synsets(pos='n')):
    #         # write the transitive closure of all hypernyms of a synset to file
    #         for hyper in synset.closure(lambda s: s.hypernyms()):
    #             edges.add((synset.name(), hyper.name()))

    #         # also write transitive closure for all instances of a synset
    #         for instance in synset.instance_hyponyms():
    #             for hyper in instance.closure(lambda s: s.instance_hypernyms()):
    #                 edges.add((instance.name(), hyper.name()))
    #                 for h in hyper.closure(lambda s: s.hypernyms()):
    #                     edges.add((instance.name(), h.name()))

    #     self.nouns = pandas.DataFrame(list(edges), columns=['id1', 'id2'])
    #     self.nouns['weight'] = 1

    def set_concept_dictionary(self):
        with open('conceptnet/conceptnet-assertions-5.6.0.csv.en', "r", encoding="utf8") as f:
            for line in f.readlines():
                ls = line.strip().split('\t')
                rel = ls[0]
                head = ls[1]
                tail = ls[2]
                head = head.replace('_', ' ')
                head = self.wordnet_lemma.lemmatize(head)
                tail = tail.replace('_', ' ')
                tail = self.wordnet_lemma.lemmatize(tail)
                if head not in self.concept_dictionary:
                    self.concept_dictionary[head] = set()
                if tail not in self.concept_dictionary:
                    self.concept_dictionary[tail] = set()
                self.concept_dictionary[head].add(tail)
                self.concept_dictionary[tail].add(head)
                # if len(tail.split(' ')) > 1:
                #     for tail_split in tail.split(' '):
                #         self.concept_dictionary[head].add(tail_split)

    def compute(self, caption, objects, with_tag=False):
        object_preds = objects
        spacy_caption = self.nlp(caption)

        N_concepts = []
        for token in spacy_caption:
            if token.is_alpha and not token.is_stop:
                N_concepts.append(token.text)

        match_final_dict = dict()


        for tagid in objects:
            if with_tag:
                tag = tagid
            else:
                tag = self.objects_vocab[tagid]
            tag = tag.split()[-1]
            # print(tag)
            lem_tag = self.wordnet_lemma.lemmatize(tag)
            
            lem_tag_singular = wordnet.morphy(lem_tag)
            if lem_tag_singular is None:
                lem_tag_singular = lem_tag

            tag_synonyms = set()
            for candidate in wordnet.synsets(lem_tag_singular):
                cand = self.wordnet_lemma.lemmatize(candidate.name().split('.')[0])
                cand_singular = wordnet.morphy(cand)
                tag_synonyms.add(cand_singular)

            for token in N_concepts:
                
                
                result = []
                lem_token = self.wordnet_lemma.lemmatize(token)
                lem_token_singular = wordnet.morphy(lem_token)
                if lem_token_singular is None:
                    lem_token_singular = lem_token

                # exact matching
                if lem_tag_singular == lem_token_singular:
                    if tag not in match_final_dict:
                        match_final_dict[tag] = set()
                    match_final_dict[tag].add(token)
                    result.append("exact matching")

                # plurar - singular
                singular_tag = p.singular_noun(lem_tag_singular)
                singular_token = p.singular_noun(lem_token_singular)
                if singular_tag:
                    if singular_tag == lem_token_singular:
                        if tag not in match_final_dict:
                            match_final_dict[tag] = set()
                        match_final_dict[tag].add(token)
                        result.append("plurar,singular")
                if singular_token:
                    if singular_token == lem_tag_singular:
                        if tag not in match_final_dict:
                            match_final_dict[tag] = set()
                        match_final_dict[tag].add(token)
                        result.append("plurar,singular")

                # concept matching
                if lem_token_singular in self.concept_dictionary:
                    for concept in self.concept_dictionary[lem_token_singular]:
                        if lem_tag == concept or lem_tag_singular == concept:
                            if tag not in match_final_dict:
                                match_final_dict[tag] = set()
                            match_final_dict[tag].add(token)
                            result.append("concept")
                            # result.append(self.concept_dict_rel[lem_token_singular])
                            # if token.lower() == 'sleeping' and tag.lower() == 'person':
                            #     print(lem_token_singular, concept)

                # word vector similarity
                if lem_token_singular not in self.gensim.vocab or lem_tag_singular not in self.gensim.vocab:
                    continue
                
                if self.gensim.similarity(lem_tag_singular, lem_token_singular) > 0.65:
                    if tag not in match_final_dict:
                        match_final_dict[tag] = set()
                    match_final_dict[tag].add(token)
                    result.append("word2vec")

                # synonyms
                if lem_token_singular in tag_synonyms:
                    if tag not in match_final_dict:
                        match_final_dict[tag] = set()
                    match_final_dict[tag].add(token)
                    result.append("synonym")

                # if tag.lower() == 'office' and token.lower() == 'computer':
                #     print(result)

                # hypernym / hyponym
                # token_syn = None
                # syns = wordnet.synsets(lem_token_singular)
                # if len(syns) > 0:
                #     token_syn = wordnet.synsets(lem_token_singular)[0]
                #
                # if token_syn is not None:
                #     # as_hypernym_set = set(nouns[nouns.id2 == token_syn.name()].id1.unique())
                #     # as_hypernym_set.add(token_syn.name())
                #     #
                #     # hypo_hypernym = nouns[nouns.id1.isin(as_hypernym_set) & nouns.id2.isin(as_hypernym_set)]
                #     final_candidate_set = set()
                #
                #     # for candidate in list(set(hypo_hypernym['id1'].unique())):
                #     #     if candidate.split('.')[-1] == '01':
                #     #         cand = wordnet_lemmatizer.lemmatize(candidate.split('.')[0])
                #     #         cand_singular = wordnet.morphy(cand)
                #     #         final_candidate_set.add(cand_singular)
                #     # for candidate in list(set(hypo_hypernym['id2'].unique())):
                #     #     if candidate.split('.')[-1] == '01':
                #     #         cand = wordnet_lemmatizer.lemmatize(candidate.split('.')[0])
                #     #         cand_singular = wordnet.morphy(cand)
                #     #         final_candidate_set.add(cand_singular)
                #     for candidate_path in wordnet.synset(token_syn.name()).hypernym_paths():
                #         for candidate in candidate_path:
                #             if candidate.name().split('.')[-1] == '01':
                #                 cand = self.wordnet_lemma.lemmatize(candidate.name().split('.')[0])
                #                 cand_singular = wordnet.morphy(cand)
                #                 final_candidate_set.add(cand_singular)
                #
                #     if tag in final_candidate_set:
                #         if tag not in match_final_dict[b_int]:
                #             match_final_dict[b_int][tag] = set()
                #         match_final_dict[b_int][tag].add(token)

        return match_final_dict
