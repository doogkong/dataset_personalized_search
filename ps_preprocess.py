import pandas as pd
import logging 
import sys 
import time
import json
import os 
import argparse
from ps_utils import QREL, CTX_MAP, CTX_MAP_RVS

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger=logging.getLogger()

def build_qmap(query):
    q_map, q_map_rvs = dict(), dict()
    ctx = json.load(open(query))
    for item in ctx: 
        id, val = item['query_id'], item['query'].strip()
        q_map[id] = val 
        q_map_rvs[val] = id
    return q_map, q_map_rvs 

def build_url2doc(doc_corpus):
    map_doc2url, map_url2doc = dict(), dict()
    cnt = 0
    with open(doc_corpus, 'r') as fin:
        for line in fin.readlines():
            item = line.split(",")
            doc_id, url = item[0], item[1].strip().lower()
            cnt += 1 
            map_doc2url[doc_id] = url 
            map_url2doc[url] = doc_id
    return map_doc2url, map_url2doc 


def post_process(qrels_in, qrels_out):
    """
    Post-processing to remove the duplicate ones. 
    :param qrels_input: qrels input 
    :param qrels_output: qrels output after removing duplicate ones
    """
    fp_in = open(qrels_in, 'r')
    fp_out= open(qrels_out, 'w')
    qrels_df = pd.read_csv(qrels_in, sep=' ', header=None, \
    names=['qid', 'unused', 'doc_id', 'label', 'job', 'geo'], dtype={'query_id':'str', 'unused':'str',\
     'doc_id':'str', 'label':'str'})
    qrels_df["label"] = qrels_df["label"].astype('int32')
    qrels_df["job"] = qrels_df["job"].astype('int32')
    qrels_df["geo"] = qrels_df["geo"].astype('int32')
    DUP_MAP = {}
    for index, row in qrels_df.iterrows():
        qid = row['qid']
        doc_id = row['doc_id']
        key = str(qid) + " Q0 " + str(doc_id)
        cur_line = key + " " +str(row["label"]) + " " + str(row["job"]) + " " + str(row["geo"])
        if key in DUP_MAP:
            DUP_MAP[key].append(cur_line)
        else: 
            DUP_MAP[key] = [cur_line]
    for key in DUP_MAP:
        line = DUP_MAP[key][0]
        fp_out.write(str(line) + "\n")
    fp_in.close()
    fp_out.close()


def gen_qrels_add_context(infile, qmap_rvs, map_url2doc, out_file):
    """
     Generate qrel file with added context using human annotations 
     :param infile: human annotation 
     :param qmap_rvs: query file map
     :param map_url2doc: doc_corpus_path
     :param out_file: output as qrels_path
     """
    start = time.time()
    df = pd.read_csv(infile, sep=',', usecols = ['did','query','job','geo','url','score'])
    fout = open(out_file, 'w')
    HIT = 0
   
    for index, row in df.iterrows():
        query = row['query'].strip()
        url = row['url'].strip().lower()
        job, geo = row['job'], row['geo']
        
        job_id, geo_id = CTX_MAP.get(job, 0), CTX_MAP.get(geo, 0)
        doc_id = map_url2doc.get(url, -1)
        HIT = 1 if doc_id != -1 else 0 
        score = QREL[row['score'].upper()]
        logger.info("index={0},query={1},qid={2}, job={3}, location={4}, url={5}, score={6}, val={7}, hit={8}"\
                .format(index, row['query'], \
                qmap_rvs[row['query'].strip()], \
                row['job'], row['geo'], row['url'], row['score'],\
                QREL[row['score'].upper()] if row['score'] is not None else 0, HIT))

        if HIT == 1 and score > 0:
            line = str(qmap_rvs[row['query'].strip()]) \
                + " "+ str("Q0") + " " \
                + str(doc_id) + " "  \
                + str(QREL[row['score'].upper()]) + " "\
                + str(job_id) + " "\
                + str(geo_id) + "\n"
            fout.write(line)
    logger.info('read_qrels: {:.03f}'.format(time.time() - start))


if __name__ == "__main__":
    """
    Purpose: Generate qrel file with added context using 90k human annotations given queries and doc corpus 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries_path', required=True)         
    parser.add_argument('--human_annotation_path', required=True)  
    parser.add_argument('--doc_corpus_path', required=True)
    parser.add_argument('--output_qrels_path', required=True)

    args = parser.parse_args()
    logger.info('args: %s', args)
    
    logger.info("Step 1: Loading queries ...............")
    query = args.queries_path    
    qmap, qmap_rvs = build_qmap(query)
    logger.info(list(qmap.items())[0:10])

    logger.info("Step 2: Loading human annotations and doc_corpus ...............")
    human_annotation = args.human_annotation_path
    doc_corpus = args.doc_corpus_path
    map_doc2url, map_url2doc  = build_url2doc(doc_corpus)
    logger.info(list(map_url2doc.items())[0:10])
    
    logger.info("Step 3: Generating qrel files ...............")
    output_qrels = args.output_qrels_path
    output_qrels_tmp = output_qrels + ".tmp"
    gen_qrels_add_context(human_annotation, qmap_rvs, map_url2doc, output_qrels_tmp)
    post_process(output_qrels_tmp, output_qrels)
    
    """
    # export PYTHONPATH=`pwd`:$PYTHONPATH
    # python ps_preprocess.py --queries_path queries.json --human_annotation_path dr_90k_annotation.csv  --doc_corpus_path doc2url.csv --output_qrels_path qrels_dr_90k.txt
    export in_queryfile=$data_dir/queries.json
    export in_doc_corpus_path=$data_dir/doc2url.csv
    export in_human_annotation=$data_dir/dr_90k_annotation.csv
    export output_qrels_path=$data_dir/qrels_dr_90k.txt
    export PYTHONPATH=`pwd`:$PYTHONPATH
    python ps_preprocess.py --queries_path=$in_queryfile --human_annotation_path=$in_human_annotation --doc_corpus_path=$in_doc_corpus_path --output_qrels_path=$output_qrels_path
    """
