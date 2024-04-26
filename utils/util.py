import os
import jsonlines
import pickle
import numpy as np


def get_retrieved_pseudo_query(pq_as_query_dict, res_q2pq_dict, res_jsonl_path, pkl_path, res_tsv_path):
    retrieved_pq_dict={}
    pqidset=set()
    for pqids in res_q2pq_dict.values():
        pqidset=pqidset.union(set(pqids))
    for pqid in pqidset:
        retrieved_pq_dict[pqid]=pq_as_query_dict[pqid]
    
    texts=[]
    embeddings=[]
    ids=[]
    with jsonlines.open(res_jsonl_path,"a") as fa_jsonl:  
        for k,v in retrieved_pq_dict.items():
            content, vector = v
            res_dict={'id':k, 'contents': content, 'vector':vector}
            fa_jsonl.write(res_dict) 
            texts.append(content)
            embeddings.append(vector)  
            ids.append(k)
    assert len(ids)==len(texts) and len(ids)==len(embeddings)
    with open(pkl_path,"wb") as fa_pkl:        
        dict={'text':np.array(texts), 'embedding':np.array(embeddings)}
        pickle.dump(dict,fa_pkl)
    with open(res_tsv_path,'a') as f:
        for i in range(len(texts)):
            f.write(ids[i]+'\t'+texts[i]+'\n')


def create_dict_faiss_retrieve(reslist):
    resdict={}
    for qid,hits in reslist:
        docidlist=[]
        for dsr in hits:
            docidlist.append(dsr.docid)
        resdict[qid]=docidlist
    return resdict


def create_dict_for_jsonl(path):
    dict={}
    with jsonlines.open(path,'r') as reader:
        for data in reader:
            dict[data['id']]=data['vector']
    return dict


def create_dict_for_jsonl_with_content(path):
    dict={}
    with jsonlines.open(path,'r') as reader:
        for data in reader:
            dict[data['id']]=(data['contents'], data['vector'])
    return dict


def create_dict_trec_form_no_score(path):
    res_dict={}
    same_q=[]
    q=""
    with open(path,"r") as f:
        for line in f.readlines():
            line=line.split(" ")
            qid=line[0].strip()
            pid=line[2].strip()
            if qid != q:
                if q!="":
                    res_dict[q]=same_q
                    same_q=[]
                q=qid
            same_q.append(pid)
    res_dict[q]=same_q
    return res_dict


def create_dict_trec_form_with_score(path):
    res_dict={}
    same_q={}
    q=""
    with open(path,"r") as f:
        for line in f.readlines():
            line=line.split(" ")
            qid=line[0].strip()
            pid=line[2].strip()
            score=float(line[4])
            if qid != q:
                if q!="":
                    res_dict[q]=same_q
                    same_q={}
                q=qid
            same_q[pid]=score
    res_dict[q]=same_q
    return res_dict


def create_p5_dict(q2p_dict,num_p_center):
    res={}
    for qid,pdict in q2p_dict.items():
        pidlist=[]
        for i in range(num_p_center):
            pidlist.append(list(pdict.keys())[i])
        res[qid]=pidlist
        assert len(res[qid])==num_p_center
    return res