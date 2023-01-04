import gzip
import os
import csv
from click import progressbar
from sentence_transformers import SentenceTransformer,  util, LoggingHandler, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
import logging
import pandas as pd
import torch
import numpy as np
from open_clip import tokenize as clip_tokenizer
from zipfile import ZipFile
from data.classname_and_prompt.ImageNet import templates as ImageNet_templates
from tqdm import tqdm
from torch.utils.data import DataLoader
from evaluation.linear_eval import get_linear_eval_acc

from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import logging
import sys
import os
import tarfile
from scipy.stats import pearsonr, spearmanr
from utils.captioned_imagenet import CaptionedImageNet
# from senteval import engine

PATH_TO_SENTEVAL = 'itra/evaluation/SentEval'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

autocast = torch.cuda.amp.autocast

def ms_marco(model, args):
    corpus_max_size = 0
    ### Data files
    data_folder = '/data/Datasets/msmarco-data'
    os.makedirs(data_folder, exist_ok=True)

    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    dev_queries_file = os.path.join(data_folder, 'queries.dev.small.tsv')
    qrels_filepath = os.path.join(data_folder, 'qrels.dev.tsv')

    ### Download files if needed
    if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
        tar_filepath = os.path.join(data_folder, 'collectionandqueries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download: "+tar_filepath)
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    if not os.path.exists(qrels_filepath):
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv', qrels_filepath)

    ### Load data

    corpus = {}             #Our corpus pid => passage
    dev_queries = {}        #Our dev queries. qid => query
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need

    # Load the 6980 dev queries
    with open(dev_queries_file, encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            dev_queries[qid] = query.strip()


    # Load which passages are relevant for which queries
    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split('\t')

            if qid not in dev_queries:
                continue

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()
            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)


    # Read passages
    with open(collection_filepath, encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            passage = passage

            if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
                corpus[pid] = passage.strip()



    ## Run evaluator
    logging.info("Queries: {}".format(len(dev_queries)))
    logging.info("Corpus: {}".format(len(corpus)))

    ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[10, 100],
                                                            name="msmarco dev")
    result = ir_evaluator(model)
    logging.info(f'Finished MS Marco dev evaluation, score: {result}')
    return result

    
def sts_benchmark(model, args):
    sts_dataset_path = os.path.join(args.datasets_dir, 'stsbenchmark.tsv.gz')

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

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

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
    result_dev = evaluator(model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    result_test = evaluator(model)
    logging.info(f'Finished sts-b evaluation, score: {result_dev} (dev), {result_test} (test).')
    
    return result_dev, result_test


def sts12_sts16_eval(model, args):
    PATH_TO_DATA = os.path.join(args.datasets_dir,'simcse_sts_data')
    # Set params for SentEval
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
        embeddings = model.encode(batch, show_progress_bar=False, use_pooler=False)
        return embeddings

    # Set up the tasks
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    logging.info(f'Starting SemEval on {tasks}')
    results = {}
    for task in tqdm(tasks):
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Get evaluation results
    task_names = []
    scores = []
    for task in tasks:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    return task_names, scores


def sts_coco(model, args):
    samples = []
    with open('STS_coco_val2017.csv') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score'])
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
            samples.append(inp_example)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, name='sts-coco')
    #result = model_without_ddp.evaluate(evaluator)
    result = evaluator(model)
    logging.info(f'Finished sts-coco evaluation, score: {result}')
    return result


def wiki_sections(model, args):

    dataset_path = '/data/Datasets/wikipedia-sections'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        filepath = os.path.join(dataset_path, 'wikipedia-sections-triplets.zip')
        util.http_get('https://sbert.net/datasets/wikipedia-sections-triplets.zip', filepath)
        with ZipFile(filepath, 'r') as zip:
            zip.extractall(dataset_path)

    test_examples = []
    with open(os.path.join(dataset_path, 'test.csv'), encoding="utf-8") as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            test_examples.append(InputExample(texts=[row['Sentence1'], row['Sentence2'], row['Sentence3']]))


    test_evaluator = TripletEvaluator.from_input_examples(test_examples, name='test')
    result = test_evaluator(model)
    logging.info(f'Finished wiki sections evaluation, score: {result}')
    return result


def word_evaluations(model, args):

    def cosine_similarity(a, b):
        return ((np.dot(a, b)) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))+1e-10))

    results = []
    for EVALUATION in ['rg65.csv', 'simlex999.csv', 'wordsim353.csv']:
        intrinsic_evaluation = pd.read_csv(os.path.join(args.datasets_dir, EVALUATION), index_col=None, sep=';', header=None)
        words_left= intrinsic_evaluation[0].tolist()
        words_right = intrinsic_evaluation[1].tolist()
        relatedness = intrinsic_evaluation[2].tolist()

        words_left_extended = []
        words_right_extended = []
        relatedness_extended = []
        for i in range(len(words_left)):
            for template in ImageNet_templates:
                words_left_extended.append(template.replace('{}', words_left[i]))
                words_right_extended.append(template.replace('{}', words_right[i]))
                relatedness_extended.append(relatedness[i])
        words_left= words_left_extended
        words_right = words_right_extended
        relatedness = relatedness_extended            


        words_left_embeddings = model.encode(words_left, convert_to_numpy=True, convert_to_tensor=False)
        # print(words_left_embeddings)
        # exit()
        words_right_embeddings = model.encode(words_right, convert_to_numpy=True, convert_to_tensor=False)
        cosine_similarities = [cosine_similarity(words_left_embeddings[i],words_right_embeddings[i]) for i in range(len(words_right_embeddings))]
        evaluation_score = spearmanr(cosine_similarities,relatedness)[0]
        results.append(evaluation_score)
        logging.info(f"Finished {EVALUATION.replace('.csv', '')}\tevaluation, score: {evaluation_score}")
    return results


def imagenet_linear(model, args):
    train = CaptionedImageNet(path='data/captioned_imagenet', split='train', preprocess=None)
    test = CaptionedImageNet(path='data/captioned_imagenet', split='val', preprocess=None)
       
    #train_features, train_labels = get_features(model, train, args=args)
    #test_features, test_labels = get_features(model, test, args=args)
    print(len(train.captions))
    train_features = model.encode(train.captions)
    train_labels = train.labels
    print(len(test.captions))
    test_features = model.encode(test.captions)
    test_labels = test.labels
    
    linear_acc = get_linear_eval_acc(train_features, train_labels, test_features, test_labels, args)
    return  linear_acc


def nlp_eval(model, epoch, args):
    model = model.module if args.distributed else model
    if args.nlp_eval_frequency == 0:
        return {}
    if (epoch % args.nlp_eval_frequency) != 0 and epoch != args.epochs:
        return {}

    results = {}
    result_dev, result_test = sts_benchmark(model, args)
    results['sts-b-dev'] = result_dev
    results['sts-b-test'] = result_test
    
    sts12_sts16_task_names, scores = sts12_sts16_eval(model, args)
    length = len(sts12_sts16_task_names)
    for i in range(length):
        results[sts12_sts16_task_names[i]] = scores[i]
        logging.info(f'\t{sts12_sts16_task_names[i]}\t{scores[i]}')
        
    # captioned_imagenet = imagenet_linear(model, args)
    # results['captioned_imagenet'] = captioned_imagenet
    
    # rg65, simlex999, wordsim353 = word_evaluations(model, args)
    # results['rg65'] = rg65
    # results['wordsim353'] = wordsim353
    # results['simlex999'] = simlex999
    
    # # sts_result = sts_coco(model, args)
    # # results['sts-coco'] = sts_result
    
    # wiki_section_result = wiki_sections(model, args)
    # results['wiki-sections'] = wiki_section_result   

    # ms_marcro_result = ms_marco(model, args)
    # results['ms-marco'] = ms_marcro_result   

    return results


