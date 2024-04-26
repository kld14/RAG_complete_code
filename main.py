import os
import os.path
from jsonargparse import CLI
import argparse
from utils.search import search
from utils.util import (create_dict_faiss_retrieve, create_dict_trec_form_no_score, get_retrieved_pseudo_query,
                        create_dict_for_jsonl, create_p5_dict, create_dict_trec_form_with_score, 
                        create_dict_for_jsonl_with_content)
from utils.expand_pq import expand_pq_by_cosine_similarity
from utils.calculate_score_hybrid import cal_score


def search_main(
        passage_index_path: str = '/home1/kld/RAG-plus/indexes/bge/abstracts',
        passage_embedding_path: str = '/home1/kld/RAG-plus/bge-embeddings/abstracts/used_abstracts.jsonl',
        pseudo_query_as_query_path: str = '/home1/kld/RAG-plus/bge-embeddings/pseudo_query_as_query/pseudo_query_as_query.jsonl',
        pseudo_query_as_query_dir: str = '/home1/kld/RAG-plus/bge-embeddings/pseudo_query_as_query',
        pseudo_query_as_passage_path: str = '/home1/kld/RAG-plus/bge-embeddings/pseudo_query_as_passage/pseudo_query_as_passage.jsonl',
        pseudo_query_as_passage_index_path: str = '/home1/kld/RAG-plus/indexes/bge/pseudo_query',
        encoded_query_dir: str = '/home1/kld/RAG-plus/bge-embeddings/query',
        query_embedding_path: str = '/home1/kld/RAG-plus/bge-embeddings/query/query.jsonl',
        topic: str = '/home1/kld/RAG-plus/doris-mae-queries-test.tsv',
        hit_q2pq: int = 5,
        hit_pq2pq: int = 6,
        hit_pq2p: int = 500,
        num_p_center: int = 5,
        w_query: float = 0.3,
        w_pseudo_query: float = 0.3,
        w_passage: float = 0.4,
        threshold: float = 0.2,
        weight: int = 100,
        chain: bool = False,
        query_to_passage: str = '/home1/kld/RAG-plus/baseline-res/bge/q2p_150.txt',
        dir_name: str = '/home1/kld/RAG_complete_code/pseudo_query_bge_'
):
    
    # query retrieve pseudo query
    q2pq_output_path=f'{dir_name}/q2pq_5.txt'
    if not os.path.exists(q2pq_output_path):
        parser = argparse.ArgumentParser(description='query retrieve pseudo query.')
        parser.add_argument('--topics', type=str, metavar='topic_name', required=False, default=topic,
                            help="Name of topics. Available: msmarco-passage-dev-subset.")
        parser.add_argument('--index', type=str, metavar='path to index or index name', required=False,
                            default=pseudo_query_as_passage_index_path,
                            help="Path to Faiss index or name of prebuilt index.")
        parser.add_argument('--encoded-queries', type=str, metavar='path to query encoded queries dir or queries name',
                            required=False, default=encoded_query_dir, 
                            help="Path to query encoder pytorch checkpoint or hgf encoder model name")
        parser.add_argument('--hits', type=int, metavar='num', required=False, default=hit_q2pq, help="Number of hits.")
        parser.add_argument('--output', type=str, metavar='path', required=False, default=q2pq_output_path, help="Path to output file.")
        parser.add_argument('--batch-size', type=int, metavar='num', required=False, default=36,
                            help="search batch of queries in parallel")
        parser.add_argument('--threads', type=int, metavar='num', required=False, default=1,
                            help="maximum threads to use during search")
        q2pq_output=search(parser)
        res_q2pq_dict=create_dict_faiss_retrieve(q2pq_output)
    else:
        res_q2pq_dict=create_dict_trec_form_no_score(q2pq_output_path)
    
    # pseudo query retrieve pseudo query
    pq_as_query_dict=create_dict_for_jsonl_with_content(pseudo_query_as_query_path)

    retrieved_q2expanded_pq_dir=f'{dir_name}/retrieved_{hit_q2pq}pq'
    res_jsonl_path=f'{retrieved_q2expanded_pq_dir}/retrieved_pseudo_query.jsonl'
    pkl_path=f'{retrieved_q2expanded_pq_dir}/embedding.pkl'
    res_tsv_path=f'{retrieved_q2expanded_pq_dir}/retrieved_pseudo_query.tsv'
    if not os.path.exists(retrieved_q2expanded_pq_dir):
        os.mkdir(retrieved_q2expanded_pq_dir)
        get_retrieved_pseudo_query(pq_as_query_dict, res_q2pq_dict, res_jsonl_path, pkl_path, res_tsv_path)
    
    pq2pq_output_path=f'{dir_name}/pq2pq1_{hit_pq2pq}.txt'
    if not os.path.exists(pq2pq_output_path):
        parser1 = argparse.ArgumentParser(description='pseudo query retrieve pseudo query.')
        parser1.add_argument('--topics', type=str, metavar='topic_name', required=False, default=res_tsv_path)
        parser1.add_argument('--index', type=str, metavar='path to index or index name', required=False,
                            default=pseudo_query_as_passage_index_path)
        parser1.add_argument('--encoded-queries', type=str, metavar='path to query encoded queries dir or queries name',
                            required=False, default=pseudo_query_as_query_dir)
        parser1.add_argument('--hits', type=int, metavar='num', required=False, default=hit_pq2pq)
        parser1.add_argument('--output', type=str, metavar='path', required=False, default=pq2pq_output_path)
        parser1.add_argument('--batch-size', type=int, metavar='num', required=False, default=36)
        parser1.add_argument('--threads', type=int, metavar='num', required=False, default=1)
        pq2pq_output=search(parser1)
        res_pq2pq_dict=create_dict_faiss_retrieve(pq2pq_output)
    else:
        res_pq2pq_dict=create_dict_trec_form_no_score(pq2pq_output_path)
    
    pq_as_passage_dict=create_dict_for_jsonl(pseudo_query_as_passage_path)
    query_emb_dict=create_dict_for_jsonl(query_embedding_path)
    passage_emb_dict=create_dict_for_jsonl(passage_embedding_path)
    q2ab_all_dict=create_dict_trec_form_with_score(query_to_passage)
    q2ab_dict=create_p5_dict(q2ab_all_dict,num_p_center)

    # expand pseudo query
    expanded_pq, expanded_qid = expand_pq_by_cosine_similarity(query_emb_dict, passage_emb_dict, pq_as_passage_dict, res_q2pq_dict, q2ab_dict, 
                                   res_pq2pq_dict, w_query, w_pseudo_query, w_passage, threshold)
        # expanded_pq: {qid: [(pqid, score)]}
    print(len(expanded_qid))

    # chain expand pseudo query
    
    # expanded pseudo query retrieve passage
    res_q2expanded_pq_dict={}
    for qid, pqid_score_list in expanded_pq.items():
        pqid_list=[pqid for pqid, _ in pqid_score_list]
        res_q2expanded_pq_dict[qid]=pqid_list
    
    retrieved_q2expanded_pq_dir=f'{dir_name}/retrieved_pq1_{threshold}_{w_query}_{w_pseudo_query}_{w_passage}'
    res_jsonl_path=f'{retrieved_q2expanded_pq_dir}/retrieved_pseudo_query.jsonl'
    pkl_path=f'{retrieved_q2expanded_pq_dir}/embedding.pkl'
    res_tsv_path=f'{retrieved_q2expanded_pq_dir}/retrieved_pseudo_query.tsv'
    if not os.path.exists(retrieved_q2expanded_pq_dir):
        os.mkdir(retrieved_q2expanded_pq_dir)
        get_retrieved_pseudo_query(pq_as_query_dict, res_q2expanded_pq_dict, res_jsonl_path, pkl_path, res_tsv_path)

    expanded_pq2p_path=f'{dir_name}/expanded_pq2p.txt'
    parser2 = argparse.ArgumentParser(description='expanded pseudo query retrieve passage.')
    parser2.add_argument('--topics', type=str, metavar='topic_name', required=False, default=res_tsv_path)
    parser2.add_argument('--index', type=str, metavar='path to index or index name', required=False,
                        default=passage_index_path)
    parser2.add_argument('--encoded-queries', type=str, metavar='path to query encoded queries dir or queries name',
                        required=False, default=pseudo_query_as_query_dir)
    parser2.add_argument('--hits', type=int, metavar='num', required=False, default=hit_pq2p)
    parser2.add_argument('--output', type=str, metavar='path', required=False, default=expanded_pq2p_path)
    parser2.add_argument('--batch-size', type=int, metavar='num', required=False, default=36)
    parser2.add_argument('--threads', type=int, metavar='num', required=False, default=1)
    expanded_pq2p_output=search(parser2)

    # calculate final score
    res_q2expanded_pq_dict_with_score={}
    for qid, pqid_score_list in expanded_pq.items():
        pqid_score_dict={}
        for pqid, score in pqid_score_list:
            pqid_score_dict[pqid]=score
        res_q2expanded_pq_dict_with_score[qid]=pqid_score_dict
    
    res_expanded_pq2p_dict_with_score={}
    for qid,hits in expanded_pq2p_output:
        docid_score_dict={}
        for dsr in hits:
            docid_score_dict[dsr.docid]=dsr.score
        res_expanded_pq2p_dict_with_score[qid]=docid_score_dict
    
    output_path=f'{dir_name}/q2ab_expanded_pq_{threshold}_{w_query}_{w_pseudo_query}_{w_passage}_{hit_pq2p}_weighted_{weight}.txt'
    cal_score(res_q2expanded_pq_dict_with_score, res_expanded_pq2p_dict_with_score, q2ab_all_dict, weight, output_path)


if __name__ == '__main__':
    CLI(search_main)