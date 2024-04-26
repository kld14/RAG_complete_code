from scipy.spatial.distance import cosine
from typing import List, Tuple, Dict
import jsonlines

def cal_cosine_dist(q_emb:List[float], pq_emb:List[List[float]], p_emb:List[List[float]], new_pq_emb:List[float], w1, w2, w3):
    #query, pseudo query, passage
    cos=0
    cos+=w1*cosine(q_emb,new_pq_emb)
    for embs in pq_emb:
        cos+=w2*cosine(embs,new_pq_emb)
    for embs in p_emb:
        cos+=w3*cosine(embs,new_pq_emb)
    return cos


def rerank_pq(q_emb:List[float], pq:List[Tuple[str,List[float]]]):
    res=[]
    for pqid, pqemb in pq:
        cd=cosine(q_emb,pqemb)
        cs=1-cd
        res.append((pqid,cs))
    res.sort(key=lambda x:x[1],reverse=True)
    return res


def expand_pq_by_cosine_similarity(query_emb_dict, passage_emb_dict, pq_emb_dict, q2pq_dict, q2ab_dict, 
                                   pq2pq_dict, w1, w2, w3, threshold):
    expanded_qid=[]
    res_dict={}
    for qid, pqlist in q2pq_dict.items():
        q_emb=query_emb_dict[qid]
        pq_embs=[pq_emb_dict[pq] for pq in pqlist]
        assert len(pq_embs)==5
        ablist=q2ab_dict[qid]
        ab_embs=[passage_emb_dict[p] for p in ablist]
        new_pq_list=[]
        for pq in pqlist:
            new_pq_list.extend(pq2pq_dict[pq])
        assert len(new_pq_list)==30
        new_pq_list_set=set(new_pq_list)
        new_pq_list=list(new_pq_list_set-set(pqlist))
        assert len(new_pq_list)>5
        new_pq_embs=[pq_emb_dict[pq] for pq in new_pq_list]
        new_pqs=list(zip(new_pq_list,new_pq_embs))

        filtered_new_pq=[]
        for new_pq, new_pq_emb in new_pqs:
            cosine_dist=cal_cosine_dist(q_emb,pq_embs,ab_embs,new_pq_emb,w1,w2,w3)
            cosine_similarity=1-cosine_dist
            if cosine_similarity>threshold:
                filtered_new_pq.append(new_pq)
        if len(filtered_new_pq)>0:
            expanded_qid.append(qid)
        all_pq_list=pqlist+filtered_new_pq
        all_pq_embs=[pq_emb_dict[pq] for pq in all_pq_list]
        all_pq=list(zip(all_pq_list,all_pq_embs))
        reranked_pq=rerank_pq(q_emb,all_pq) #[(pqid, score)]
        res_dict[qid]=reranked_pq
    return (res_dict, expanded_qid)


def expand_pq_chain_by_cosine_similarity(level, query_emb_dict, passage_emb_dict, pq_emb_dict, q2pq_dict, q2ab_dict, 
                                   q2exp_pq_with_score, exp_pq2pq_dict, pre_expanded_qid, w1, w2, w3, threshold):
    # q2exp_pq_with_score: {qid: [(pqid, score)]}
    expanded_qid=[]
    res_dict={}
    for qid, pqlist in q2pq_dict.items():
        if qid in pre_expanded_qid:
            q_emb=query_emb_dict[qid]
            pq_embs=[pq_emb_dict[pq] for pq in pqlist]
            assert len(pq_embs)==5
            ablist=q2ab_dict[qid]
            ab_embs=[passage_emb_dict[p] for p in ablist]
            exp_pqlist=[pqid for pqid, _ in q2exp_pq_with_score[qid]]
            new_pq_list=[]
            for pq in exp_pqlist:
                new_pq_list.extend(exp_pq2pq_dict[pq])
            new_pq_list_set=set(new_pq_list)
            new_pq_list=list(new_pq_list_set-set(exp_pqlist))
            new_pq_embs=[pq_emb_dict[pq] for pq in new_pq_list]
            new_pqs=list(zip(new_pq_list,new_pq_embs))

            filtered_new_pq=[]
            for new_pq, new_pq_emb in new_pqs:
                cosine_dist=cal_cosine_dist(q_emb,pq_embs,ab_embs,new_pq_emb,w1,w2,w3)
                cosine_similarity=1-cosine_dist
                if cosine_similarity>threshold:
                    filtered_new_pq.append(new_pq)
            if len(filtered_new_pq)>0:
                expanded_qid.append(qid)
                all_pq_list=exp_pqlist+filtered_new_pq
                all_pq_embs=[pq_emb_dict[pq] for pq in all_pq_list]
                all_pq=list(zip(all_pq_list,all_pq_embs))
                reranked_pq=rerank_pq(q_emb,all_pq) #[(pqid, score)]
                res_dict[qid]=reranked_pq
            else:
                res_dict[qid]=q2exp_pq_with_score[qid]
        else:
            res_dict[qid]=q2exp_pq_with_score[qid]
    with open(str(level)+'.tsv','a') as fa:
        for qid, hits in res_dict.items():
            for pqid,_ in hits:
                fa.write(qid+'\t'+pqid+'\n')
    return (res_dict, expanded_qid)