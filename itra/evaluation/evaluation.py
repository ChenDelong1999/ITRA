import logging
import os
import json
from training.distributed import is_master
from .linear_eval import linear_eval
from .zero_shot import zero_shot_eval
from .retrieval import retrieval_evaluation
# from .analyze_features import analyze_features
# from .sts_evaluation import sts_benchmark
from .nlp_evaluations import nlp_eval
from .wise_ft import get_wise_ft_model

try:
    import wandb
except ImportError:
    wandb = None

def evaluate(model, epoch, preprocess, args, tb_writer=None):
    if args.distributed and not is_master(args):
        return
    logging.info( f"Starting evaluation of [{args.name}] at epoch {epoch}")


    if args.eval_with_wise_ft !=1:
        logging.info( f"Perform Wise-FT evaluation with alpha={args.eval_with_wise_ft}")
        model = get_wise_ft_model(model, args, alpha=args.eval_with_wise_ft)
        distributed = args.distributed
        args.distributed = False

    if args.model_ema:
        distributed = args.distributed
        args.distributed = False
    
    linear_eval_datasets = ['CIFAR10']
    zeroshot_datasets = ['ImageNet']
    args.evaluation_workers = 8

    # zeroshot_datasets= [
    #     'ImageNet-CN',
    #     'ImageNet',
    #     'birdsnap', 
    #     'CIFAR10', 
    #     'CIFAR100', 
    #     'country211',
    #     'DTD', 
    #     'EuroSAT',
    #     'FGVCAircraft', 
    #     'flowers102', 
    #     'Food101', 
    #     'GTSRB',
    #     'MNIST', 
    #     'OxfordIIITPet', 
    #     'RenderedSST2',
    #     'StanfordCars', 
    #     'STL10',
    #     'ucf101', 
    #     #'Caltech101', 
    #     #'Flowers102', 
    #     #'SUN397', 
    #     #'CLEVER'
    #     ]
    # linear_eval_datasets= [
    #     'CIFAR10', 
    #     'CIFAR100', 
    #     'DTD', 
    #     'EuroSAT',
    #     'FGVCAircraft', 
    #     'Flowers102', 
    #     'Food101', 
    #     'GTSRB',
    #     'MNIST', 
    #     'OxfordIIITPet', 
    #     'RenderedSST2',
    #     'StanfordCars',
    #     'STL10',  
    #     'ImageNet-50k', 
    #     'ImageNet', 
    #     #'CLEVER',
    #     # 'Caltech101', 
    #     # 'SUN397', 
    #     ]
    
    model.eval()
    all_metrics = {}
    
    # NLP evaluation
    # score = sts_benchmark(model, args)
    # if tb_writer is not None:
    #     tb_writer.add_scalar(f"eval_NLP_eval/sts_benchmark", score, epoch)
    # if args.wandb:
    #     wandb.log({f"eval_NLP_eval/sts_benchmark": score, 'epoch': epoch})
    # NLP evaluation

    nlp_metrics = nlp_eval(model, epoch, args)
    all_metrics.update(nlp_metrics)
    for name, val in nlp_metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_nlp/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_nlp/{name}": val, 'epoch': epoch})
    
    # zeroshot classification
    metrics = {}
    for zeroshot_dataset in zeroshot_datasets:
        zeroshot_metrics = zero_shot_eval(model, zeroshot_dataset, epoch, preprocess, args)
        metrics.update(zeroshot_metrics)
        all_metrics.update(zeroshot_metrics)
    for name, val in metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_zero_shot/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_zero_shot/{name}": val, 'epoch': epoch})
        
    # Image-text retrieval
    metrics = {}
    retrieval_metrics = retrieval_evaluation(model, epoch, preprocess, args)
    metrics.update(retrieval_metrics)
    all_metrics.update(retrieval_metrics)
    for name, val in metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_retrieval/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_retrieval/{name}": val, 'epoch': epoch})

    # # MS-COCO retrieval
    # metrics = {}
    # retrieval_metrics, all_image_features, all_text_features= coco_retrieval_evaluation(model, epoch, preprocess, args)
    # metrics.update(retrieval_metrics)
    # all_metrics.update(retrieval_metrics)
    # for name, val in metrics.items():
    #     if tb_writer is not None:
    #         tb_writer.add_scalar(f"eval_retrieval/{name}", val, epoch)
    #     if args.wandb:
    #         wandb.log({f"eval_retrieval/{name}": val, 'epoch': epoch})

    
    # # Analyse COCO features
    # if not fast_evaluation:
    #     feature_metrics = analyze_features(all_image_features, all_text_features, args)
    #     all_metrics.update(feature_metrics)
    #     for name, val in feature_metrics.items():
    #         if tb_writer is not None:
    #             tb_writer.add_scalar(f"eval_analyze_features/{name}", val, epoch)
    #         if args.wandb:
    #             wandb.log({f"eval_analyze_features/{name}": val, 'epoch': epoch})

    # linear evaluation
    metrics = {}
    if linear_eval_datasets:
        linear_metrics = linear_eval(model, linear_eval_datasets, epoch, preprocess, args)    
        metrics.update(linear_metrics)
        all_metrics.update(linear_metrics)

    logging.info( f"Finished evaluation of [{args.name}] at epoch {epoch}\n" + "\n".join([f"\t{k}\t{v}" for k, v in all_metrics.items()]))

    for name, val in metrics.items():
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval_linear_prob/{name}", val, epoch)
        if args.wandb:
            wandb.log({f"eval_linear_prob/{name}": val, 'epoch': epoch})
                
    if args.save_logs:
        with open(os.path.join(args.logs, args.name, "results.jsonl"), "a+") as f:
            f.write(json.dumps(all_metrics))
            f.write("\n")
            
            
    if args.eval_with_wise_ft !=1 or args.model_ema:
        args.distributed = distributed
        
    return all_metrics
