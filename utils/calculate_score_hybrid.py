from scipy.special import softmax
import numpy as np
np.set_printoptions(precision=4)


def get_all_pid_list(lists):
    all_pid_list = []
    for lst in lists:
        all_pid_list += lst
    all_pid_list=set(all_pid_list)
    return list(all_pid_list)


def cal_score_for_q2p(query_pseudo, pseudo_passage):
    res_q2p_dict={}
    for qid,pq_hits in query_pseudo.items():
        pq_hit_scores_np=np.array(list(pq_hits.values()))
        sim_q_pq_softmax=softmax(pq_hit_scores_np)
        pq2p_hits=[]
        for pqid in pq_hits.keys():
            pq2p_hits.append(list(pseudo_passage[pqid].keys()))
        all_pid_list=get_all_pid_list(pq2p_hits)
        passage_scores=[]
        for _pqid in pq_hits.keys():
            plist=list(pseudo_passage[_pqid].keys()) #每个pq检索的pid
            res_pid_dict={p:0 for p in all_pid_list}
            for _pid in plist:
                if _pid in res_pid_dict:
                    res_pid_dict[_pid]=pseudo_passage[_pqid][_pid]
            passage_score=list(res_pid_dict.values())
            passage_scores.append(passage_score)
        passage_scores_np=np.array(passage_scores)
        sim_q_d_np=np.matmul(sim_q_pq_softmax,passage_scores_np)
        sim_q_d=sim_q_d_np.tolist()
        pid_scores=sorted(list(zip(sim_q_d,all_pid_list)),reverse=True) # [(score, pqid),(...)]
        pid_scores_dict={}
        for score_, pqid_ in pid_scores:
            pid_scores_dict[pqid_]=score_
        res_q2p_dict[qid]=pid_scores_dict
    return res_q2p_dict


def min_max_norm(pseudo_abstract):
    for abstract_hits in pseudo_abstract.values():
        max_,min_=0,100000
        for score in abstract_hits.values():
            if score>max_:
                max_=score
            if score<min_:
                min_=score
        norm=max_-min_
        for pid in abstract_hits.keys():
            abstract_hits[pid]=(abstract_hits[pid]-min_)/norm
    return pseudo_abstract


def cal_score(query_pseudo, pseudo_passage, query_passage, weight, output_path):
    #calculate sim'(pq,d)
    pseudo_passage=min_max_norm(pseudo_passage)
    res_q2p_dict=cal_score_for_q2p(query_pseudo,pseudo_passage)

    with open(output_path,"a") as fa:
        for qid, pq_passage_dict in res_q2p_dict.items():        
            pq_passage_dict_key=set(list(pq_passage_dict.keys()))
            query_passage_key=set(list(query_passage[qid].keys()))
            common_pid_list_q_and_pq=set(list(pq_passage_dict.keys()))
            common_pid_list_q_and_pq.intersection_update(list(query_passage[qid].keys()))
            pq_passage_dict_key_left=pq_passage_dict_key-common_pid_list_q_and_pq
            query_passage_key_left=query_passage_key-common_pid_list_q_and_pq
            res_dict={}
            for cpid in common_pid_list_q_and_pq:
                res_dict[cpid]=pq_passage_dict[cpid]+weight*query_passage[qid][cpid]
            for pqpid in pq_passage_dict_key_left:
                res_dict[pqpid]=pq_passage_dict[pqpid]
            for qpid in query_passage_key_left:
                res_dict[qpid]=query_passage[qid][qpid]*weight
            res_sorted_dict=sorted(res_dict.items(),key=lambda x: x[1], reverse=True)

            rank=1
            pid_set=set()
            for pid,score in res_sorted_dict:
                pid_set.add(pid)
                fa.write(str(qid)+' Q0 '+str(pid)+' '+str(rank)+' '+str("%.6f"%score)+' Faiss\n')
                rank+=1
                if rank==141:
                    break