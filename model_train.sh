export CUDA_HOME=/usr/local/cuda/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
feature_type=offline
customer=is_context
expt=expt

data_dir=~/kendra_data/iscontext
qrels_path=${data_dir}/${expt}/query_relevance_plus_context_mix.txt
train_feedback=${data_dir}/${expt}/offline_feedback_plus_context200.jsonl
val_feedback=${data_dir}/${expt}/offline_feedback_context66.jsonl
feature_names_context=${data_dir}/${expt}/feature_names_context.txt
eval=in_domain_w_ctx

case $eval in 
    "in_domain_w_ctx") 
        test_feedback=${data_dir}/${expt}/offline_feedback_context66.jsonl 
        ;;
    "in_domain_wo_ctx")
        test_feedback=${data_dir}/${expt}/offline_feedback_test.jsonl
        ;;
    "out_of_domain_w_ctx")
        test_feedback=${data_dir}/${expt}/offline_feedback_context_ms_test2193.jsonl
        ;;
    "out_of_domain_wo_ctx")
        test_feedback=${data_dir}/${expt}/offline_feedback_lm.jsonl
        ;;
esac

echo $test_feedback

python DR/drpr/run_train.py --meta_only \
--dr_batch_size 64 \
--dr_n_pos 10 --dr_n_neg 5 \
--lr  0.001 \
--warmup 0.05 \
--eval_iter 100 \
--gpu 0 \
--dr_qrels $qrels_path \
--train_feedback $train_feedback \
--val_feedback $val_feedback \
--test_feedback $test_feedback \
--feat_names_file $feature_names_context \
--output_dir ${customer}_context \
--log ${customer}_context_log
