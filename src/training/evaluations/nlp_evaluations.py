import gzip
import os
import csv
from sentence_transformers import SentenceTransformer,  util, LoggingHandler, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
import logging
import pandas as pd
import torch
import numpy as np
from open_clip import tokenize as clip_tokenizer
from zipfile import ZipFile
from .openai_templets.ImageNet import templates as ImageNet_templates

from scipy.stats import pearsonr, spearmanr

autocast = torch.cuda.amp.autocast


def sts_benchmark(model, args):
    sts_dataset_path = os.path.join(args.eval_data_dir, 'stsbenchmark.tsv.gz')

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

    # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
    # model.evaluate(evaluator)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    #result = model_without_ddp.evaluate(evaluator)
    result = evaluator(model)
    logging.info(f'Finished sts evaluation, score: {result}')
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
        return ((np.dot(a, b)) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))))


    results = []
    for EVALUATION in ['rg65.csv', 'simlex999.csv', 'wordsim353.csv']:
        intrinsic_evaluation = pd.read_csv(os.path.join(args.eval_data_dir, EVALUATION), index_col=None, sep=';', header=None)
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


def nlp_eval(model, epoch, args):
    model = model.module if args.distributed else model
    if args.nlp_eval_frequency == 0:
        return {}
    if (epoch % args.nlp_eval_frequency) != 0 and epoch != args.epochs:
        return {}

    results = {}
    sts_result = sts_benchmark(model, args)
    results['sts-benchmark'] = sts_result

    if not args.fast_evaluation:
        rg65, simlex999, wordsim353 = word_evaluations(model, args)
        results['rg65'] = rg65
        results['wordsim353'] = wordsim353
        results['simlex999'] = simlex999
        
        wiki_section_result = wiki_sections(model, args)
        results['wiki-sections'] = wiki_section_result   

    return results


