"""
https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark
http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark

This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset
Usage:
python evaluation_stsbenchmark.py
OR
python evaluation_stsbenchmark.py model_name
"""
from sentence_transformers import SentenceTransformer,  util, LoggingHandler, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import sys
import torch
import gzip
import os
import csv
import argparse

# script_folder_path = os.path.dirname(os.path.realpath(__file__))

# #Limit torch to 4 threads
# torch.set_num_threads(4)

# #### Just some code to print debug information to stdout
# #### /print debug information to stdout

# model_name = sys.argv[1] if len(sys.argv) > 1 else 'all-mpnet-base-v2'

# # Load a named sentence model (based on BERT). This will download the model from our server.
# # Alternatively, you can also pass a filepath to SentenceTransformer()
# model = SentenceTransformer(model_name)



def sts_benchmark(model, args):
    sts_dataset_path = 'data/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    model_without_ddp = model.module if args.distributed else model
    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)

    # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
    # model.evaluate(evaluator)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    #result = model_without_ddp.evaluate(evaluator)
    result = evaluator(model_without_ddp)
    logging.info(f'Finished STS-benchmark evaluation, score: {result}')
    return result
