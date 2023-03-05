import pandas as pd
import logging 
import sys 
import time
import json
import argparse
from ps_utils import QREL, CTX_MAP, CTX_MAP_RVS
from DR import constants
from DR.es.search import ElasticSearchProvider
from DR.metrics import trec_eval
import os 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger=logging.getLogger()

def gen_ctx2doc_aes_score(aes_results):
    """
    Generate context to doc relevance score and rank
    :param query: query file  
    :return CTX_ES_SCORE: map of (context_id, doc_id) to its relevance score and ranks 
    """  
    CTX_ES_SCORE = dict()
    for item in aes_results:
        context_id = str(item['query_id'][1:])
        context_val = "{}".format(CTX_MAP_RVS[context_id]) 
        if context_val is not dict:
            context_val = dict()
            context_val['name'] = "{}".format(CTX_MAP_RVS[context_id]) 
        for doc in item['documents']:
            doc_id, score, rank = doc['id'], doc['score'], doc['rank']
            logger.info("context_id={}, doc_id={}, score={}, rank={}".format(context_id, doc_id, score,rank))
            context_val[doc_id] = score
            CTX_ES_SCORE[(context_id, doc_id)] = (score, rank)
    return CTX_ES_SCORE

def gen_qrel_aes_score(query, aes_run_file, es_index, flag):
    """
    Generate AE file with added context using human annotations 
     :param query: query file  
     :param aes_run_file: AES_run_file over the query files
     :param es_index:  es_index for doc corpus 
     :param flag: "non-ctx" vs. ctx indicating whether query is context or not 
     :return aes_results: AES run result 
    """     
    queries = json.load(open(query))
    aes_config="dev"
    host, port = constants.es_config(aes_config)
    id_field = "id"
    customer = "is2"
    index_version="pb2a"
    # ----------------------------------------------------------------------------
    # YOUR index for AES retrieval: 
    # $python DR/es/index.py   --index_name "$customer"_docs_pb2a_context  --corpus_path $corpus_path
    # es_index = "{}_docs_{}_context_only".format(customer, index_version)
    # ----------------------------------------------------------------------------
    query_template_path = "DR/es/config/template/doc_{}.json".format(index_version)
    query_template = open(query_template_path).read()
    es_client = ElasticSearchProvider(host,
                                   port,
                                   index=es_index,
                                   query_template=query_template,
                                   id_field=id_field)
    MAXSIZE = 10000 if flag == "ctx" else 500 
    aes_results = es_client.execute_queries(queries,size=MAXSIZE)
    trec_eval.create_runfile(aes_results, outfile=aes_run_file)

    if flag == "ctx":
        CTX_ES_SCORE = gen_ctx2doc_aes_score(aes_results)
        return CTX_ES_SCORE
    return aes_results


def read_qrel_aes_query(aes_run_file, qrels_path, queries_path):
    """
    Read qrel, aes and query files
    """
    # load ES result
    aes_df = pd.read_csv(aes_run_file, sep=' ', header=None, names=['query_id', 'unused', 'doc_id', 'rank', 'score', 'run'], \
        dtype={'query_id':'str', 'doc_id':'str'})
    # load qrels 
    qrels_df = pd.read_csv(qrels_path, sep=' ', header=None, names=['query_id', 'unused', 'doc_id', 'label', 'job', 'geo'], \
        dtype={'query_id':'str', 'doc_id':'str'})
    qrels_df["label"] = qrels_df["label"].astype('int32')
    qrels_df["job"] = qrels_df["job"].astype('int32')
    qrels_df["geo"] = qrels_df["geo"].astype('int32')
    # load query 
    queries = json.load(open(queries_path))
    return aes_df, qrels_df, queries 

def gen_qrels_pair(qrels_df):
    """
    Generate query_doc_id maps 
    """
    query_doc_ids = {}
    for index, row in qrels_df.iterrows():
        query_id, doc_id = row["query_id"], row["doc_id"]
        label, job, geo = row["label"], row["job"], row["geo"]
        if query_id not in query_doc_ids:
            query_doc_ids[query_id] = {} 
            query_doc_ids[query_id][doc_id] = [label, job, geo]
        else:
            query_doc_ids[query_id][doc_id] = [label, job, geo]
    return query_doc_ids

def write_to_feature_file(feature_file, aes_df, query_doc_ids, CTX_ES_SCORE):
    """
    Generate feature_file given the input of 1) aes_score 2) query_doc_ids 3) context score
    :param feature_file (output): generated feature file  
    :param aes_df (input): AES_file
    :param query_doc_ids (input): (query, doc) -> [label, job, geo]
    :param CTX_ES_SCORE (input): relevance scores for (job, doc) and (geo, doc)
    """
    fp_out = open(feature_file, 'w')
    for index, row in aes_df.iterrows():
        query_id, doc_id = row["query_id"], row["doc_id"]
        score, rank = row["score"], row["rank"]
        
        if query_id in query_doc_ids and doc_id in query_doc_ids[query_id]:
            label, job, geo = query_doc_ids[query_id][doc_id][0], \
                query_doc_ids[query_id][doc_id][1], query_doc_ids[query_id][doc_id][2]
        else:
            label, job, geo = 0, 0, 0 
        item = {}
        item['query_id'],  item['doc_id'] = query_id, doc_id 
        item['label'], item['job'], item['geo'] = label, job, geo
        fs = {}
        fs['es.score'], fs['es.rank'] = score, rank
        job, geo, doc_id = str(job), str(geo), str(doc_id)
        fs['user.job'], fs['user.geo'] = job, geo
        fs['doc.jobmatch_score'] = CTX_ES_SCORE[(job, doc_id)][0] if (job, doc_id) in CTX_ES_SCORE else 0 
        fs['doc.jobmatch_rank'] = CTX_ES_SCORE[(job, doc_id)][1] if (job, doc_id) in CTX_ES_SCORE else 10000 
        fs['doc.geomatch_score'] = CTX_ES_SCORE[(geo, doc_id)][0] if (geo, doc_id) in CTX_ES_SCORE else 0 
        fs['doc.geomatch_rank'] =  CTX_ES_SCORE[(geo, doc_id)][1] if (geo, doc_id) in CTX_ES_SCORE else 10000
        
        if label > 0:
            logger.info("ind={0}, index={7},query_id={8}, doc_id={9}, score={10},\
                job={1}, geo={2}, job_score={3}, job_rank={4}, geo_score={5}, geo_rank={6}"\
                .format(index, job, geo, fs['doc.jobmatch_score'], \
                 fs['doc.jobmatch_rank'], fs['doc.geomatch_score'], fs['doc.geomatch_rank'], \
                index, query_id, doc_id, score,\
                ))

        item['features'] = fs
        fp_out.write(json.dumps(item) + "\n")
    fp_out.close()


def read_feature_file(feature_file):
    """
    Read feature_file into data structure
    """
    features_by_ids = {}
    with open(feature_file, 'r') as fp:
        for line in fp:
            item = json.loads(line)
            query_id = item['query_id']
            doc_id = item['doc_id']
            item_features = item['features']
            label = item['label']
            if query_id not in features_by_ids:
                features_by_ids[query_id] = {}
            if doc_id not in features_by_ids[query_id]:
                features_by_ids[query_id][doc_id] = item_features
    return features_by_ids


def write_to_feedback_file(feedback_file, aes_result, features_by_ids, query_doc_ids):
    """
    Write 1) query_doc features, 2) (query, doc) relevance score to feedback_files
    :param aes_result (input): aes result
    :param features_by_ids (input): feature files given by (query_id, doc_id)
    :param query_doc_ids (input): label given by (query_id, doc_id)
    :param feedback_file (output): generated feedback file  
    """
    fp_out = open(feedback_file, 'w')
    for item in aes_result:
        query_id = item['query_id']
        for doc in item['documents']:
            doc_id = doc['id']
            if features_by_ids and query_id in features_by_ids and doc_id in features_by_ids[query_id]: 
                doc['features'] = features_by_ids[query_id][doc_id]
            else:
                doc['features'] = {}
            if query_doc_ids and query_id in query_doc_ids and doc_id in query_doc_ids[query_id]:
                doc['label'] = query_doc_ids[query_id][doc_id][0]
            else:
                doc['label'] = 0 
        fp_out.write(json.dumps(item) + "\n")


def gen_offline_feedback_wctx(aes_result, aes_run_file, \
    feature_file, feedback_file, qrels_path, queries_path, CTX_ES_SCORE):
    """
    Generate offline feedback/feature files using 1) qrel 2) query 3) aes_result and aes_run_file 4) context score: CTX_ES_SCORE
     :param queries_path (input): query file  
     :param aes_run_file (input): AES_run_file over the query files
     :param aes_result (input):  aes_result over the query file  
     :param qrels_path (input): (query, doc) relevance score  
     :param CTX_ES_SCORE (input): context score
     :param feature_file (output): generated feature files
     :param feedback_file (output): generated feedback files
    """     
    # load AES, qrels, query 
    aes_df, qrels_df, queries = read_qrel_aes_query(aes_run_file, qrels_path, queries_path)
    logger.info(aes_df.head())
    logger.info(qrels_df.head())
    logger.info(queries[:5])

    # query_doc_ids[query_id][doc_id] = [label, job, geo]
    query_doc_ids = gen_qrels_pair(qrels_df)
    write_to_feature_file(feature_file, aes_df, query_doc_ids, CTX_ES_SCORE)
    features_by_ids = read_feature_file(feature_file)
    logger.info("writing to feedback files ......................")
    write_to_feedback_file(feedback_file, aes_result, features_by_ids, query_doc_ids)


if __name__ == "__main__":
    """
    Given Input 1) query file, 2) index (using corpus), 3) contextual file, 4) qrels_labels 
    Generate output 1) aes_runfile, 2)offline_feature file w/ context, 3) offline_feedback files w/ context
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_query_path', required=True)
    parser.add_argument('--in_es_index', required=True)         
    parser.add_argument('--out_aes_run_file', required=True)  
    parser.add_argument('--in_ctx', required=True)         
    parser.add_argument('--out_ctx_aes_run_file', required=True)  
    parser.add_argument('--in_qrels', required=True)
    parser.add_argument('--final_out_feature_file', required=True)
    parser.add_argument('--final_out_feedback_file', required=True)


    args = parser.parse_args()
    logger.info('args: %s', args)
    logger.info("Step 1: Creating AES score for queries ...............")
    in_query = args.in_query_path    
    out_aes_run_file = args.out_aes_run_file
    in_es_index = args.in_es_index 
    out_aes_result = gen_qrel_aes_score(in_query, out_aes_run_file, in_es_index, "non-ctx")
    
    logger.info("Step 2: Create Ctx2doc AES relevance for contextual signals ...............")
    in_ctx = args.in_ctx 
    out_ctx_aes_run_file = args.out_ctx_aes_run_file
    out_CTX_ES_SCORE = gen_qrel_aes_score(in_ctx, out_ctx_aes_run_file, in_es_index, "ctx")


    logger.info("Step 3: Generating Feedback file w/ AES & Ctx signals...............")
    in_qrels = args.in_qrels 
    final_out_feature_file = args.final_out_feature_file
    final_out_feedback_file = args.final_out_feedback_file
    gen_offline_feedback_wctx(out_aes_result, \
            out_aes_run_file, \
            final_out_feature_file, final_out_feedback_file, \
            in_qrels, in_query, out_CTX_ES_SCORE)

    """
    export in_query_path=$path/queries.json
    export in_es_index=docs_pb2a_context_only
    export out_aes_run_file=$path/aes_runfile.txt
    export in_ctx=$path/contexts_jobgeo.json
    export out_ctx_aes_run_file=$path/aes_ctx_jobgeo_runfile.txt
    export in_qrels=$path/query_relevance_dr_90k_context.txt
    export final_out_feature_file=$path/offline_feature_ctx.jsonl
    export final_out_feedback_file=$path/offline_feedback_ctx.jsonl
    
    export PYTHONPATH=`pwd`:$PYTHONPATH
    python  ps_create_feedback.py \
        --in_query_path=$in_query_path \
        --in_es_index=$in_es_index \
        --out_aes_run_file=$out_aes_run_file \
        --in_ctx=$in_ctx \
        --out_ctx_aes_run_file=$out_ctx_aes_run_file \
        --in_qrels=$in_qrels \
        --final_out_feature_file=$final_out_feature_file \
        --final_out_feedback_file=$final_out_feedback_file 
    """
