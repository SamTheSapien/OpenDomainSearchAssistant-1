from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import TransportError
import json
import numpy as np

from pandas import json_normalize
import pandas as pd

import rank_metric as metrics

import sys
import os

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    ROOT = '/content/ProjectoRI2020'
else:
    ROOT = '.'


class ConvSearchEvaluation:

    def __init__(self):

        # Training topics
        with open(os.path.join(ROOT, "data/training/train_topics_v1.0.json"), "rt", encoding="utf-8") as f:
            self.train_topics = json.load(f)

        # fields: topic_turn_id, docid, rel
        self.relevance_judgments = pd.read_csv(os.path.join(ROOT, "data/training/train_topics_mod.qrel"), sep=' ',
                                               names=["topic_turn_id", "dummy", "docid", "rel"])

        # Test topics
        with open(os.path.join(ROOT, "data/evaluation/evaluation_topics_v1.0.json"), "rt", encoding="utf-8") as f:
            self.test_topics = json.load(f)

        # fields: topic_turn_id, docid, rel
        self.test_relevance_judgments = pd.read_csv(os.path.join(ROOT, "data/evaluation/evaluation_topics_mod.qrel"),
                                                    sep=' ',
                                                    names=["topic_turn_id", "dummy", "docid", "rel"])

        set_of_conversations = set(self.relevance_judgments['topic_turn_id'])
        self.judged_conversations = np.unique([a.split('_', 1)[0] for a in set_of_conversations])

    def eval(self, result, topic_turn_id):

        _precisionV = []
        _recallV=[]
        total_retrieved_docs = len(result)

        # Try to get the relevance judgments from the TRAINING set
        aux = self.relevance_judgments.loc[self.relevance_judgments['topic_turn_id'] == (topic_turn_id)]

        # IF fail, try to get the relevance judgments from the TEST set
        if np.size(aux) == 0:
            aux = self.test_relevance_judgments.loc[self.test_relevance_judgments['topic_turn_id'] == (topic_turn_id)]

        rel_docs = aux.loc[aux['rel'] != 0]

        query_rel_docs = rel_docs['docid']
        relv_judg_list = rel_docs['rel']
        total_relevant = relv_judg_list.count()

        # P@10
        top10 = result['_id'][:10]
        true_pos = np.intersect1d(top10, query_rel_docs)
        p10 = np.size(true_pos) / 10

        true_pos = np.intersect1d(result['_id'], query_rel_docs)
        recall = np.size(true_pos) / total_relevant

        #codigo nosso
        size_truePos = np.size(true_pos)
        first= np.intersect1d(result['_id'][:1], query_rel_docs)
        recallV = [0,0,0,0,0,0,0,0,0,0,0]
        p10Veval = [0,0,0,0,0,0,0,0,0,0,0]
        if size_truePos!=0:
            recallV[0] = np.size(first) / size_truePos
            p10Veval[0] = np.size(first)
            for kk in range(1, 11):
                true_pos2 = np.intersect1d(result['_id'][:10 * kk], query_rel_docs)
                recallV[kk] = np.size(true_pos2) / size_truePos
                p10Veval[kk] = np.size(true_pos2) / (10 * kk)
        # Compute vector of results with corresponding relevance level
        relev_judg_results = np.zeros((total_retrieved_docs, 1))
        for index, doc in rel_docs.iterrows():
            relev_judg_results = relev_judg_results + ((result['_id'] == doc.docid) * doc.rel).to_numpy()

        # Normalized Discount Cummulative Gain
        p10 = metrics.precision_at_k(relev_judg_results[0], 10)
        ndcg5 = metrics.ndcg_at_k(r=relev_judg_results[0], k=5, method=1)
        ap = metrics.average_precision(relev_judg_results[0], total_relevant)
        mrr = metrics.mean_reciprocal_rank(relev_judg_results[0])

        # print("Prec@10: ", p10)
        # print("NDCG@5: ", ndcg5)
        # print("AP: ", ap)
        # print("MRR: ", mrr)

        return [p10, recall, ap, ndcg5, recallV, p10Veval]

    def eval2(self, result, topic_turn_id):

        _precisionV = []
        _recallV=[]
        total_retrieved_docs = len(result)

        # Try to get the relevance judgments from the TRAINING set
        aux = self.relevance_judgments.loc[self.relevance_judgments['topic_turn_id'] == (topic_turn_id)]

        # IF fail, try to get the relevance judgments from the TEST set
        if np.size(aux) == 0:
            aux = self.test_relevance_judgments.loc[self.test_relevance_judgments['topic_turn_id'] == (topic_turn_id)]

        rel_docs = aux.loc[aux['rel'] != 0]

        query_rel_docs = rel_docs['docid']
        relv_judg_list = rel_docs['rel']
        total_relevant = relv_judg_list.count()

        # P@10
        top10 = result[:,0][:10]
        true_pos = np.intersect1d(top10, query_rel_docs)
        p10 = np.size(true_pos) / 10

        true_pos = np.intersect1d(result[:,0], query_rel_docs)
        recall = np.size(true_pos) / total_relevant

        #codigo nosso
        size_truePos = np.size(true_pos)
        first= np.intersect1d(result[:,0][:1], query_rel_docs)
        recallV = [0,0,0,0,0,0,0,0,0,0,0]
        p10Veval = [0,0,0,0,0,0,0,0,0,0,0]
        if size_truePos!=0:
            recallV[0] = np.size(first) / size_truePos
            p10Veval[0] = np.size(first)
            for kk in range(1, 11):
                true_pos2 = np.intersect1d(result[:,0][:10 * kk], query_rel_docs)
                recallV[kk] = np.size(true_pos2) / size_truePos
                p10Veval[kk] = np.size(true_pos2) / (10 * kk)
        # Compute vector of results with corresponding relevance level
        relev_judg_results = np.zeros((total_retrieved_docs, 1))
        for index, doc in rel_docs.iterrows():
            relev_judg_results = relev_judg_results + ((result[:,0] == doc.docid) * doc.rel)

        # Normalized Discount Cummulative Gain
        p10 = metrics.precision_at_k(relev_judg_results[0], 10)
        ndcg5 = metrics.ndcg_at_k(r=relev_judg_results[0], k=5, method=1)
        ap = metrics.average_precision(relev_judg_results[0], total_relevant)
        mrr = metrics.mean_reciprocal_rank(relev_judg_results[0])

        # print("Prec@10: ", p10)
        # print("NDCG@5: ", ndcg5)
        # print("AP: ", ap)
        # print("MRR: ", mrr)

        return [p10, recall, ap, ndcg5, recallV, p10Veval,relev_judg_results ]
