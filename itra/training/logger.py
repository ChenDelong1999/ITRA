import logging
import os
import time
from datetime import datetime


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} [%(filename)s:%(lineno)d] %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | [%(filename)s-%(lineno)d]: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def get_exp_name(args):
    if args.name is None:
        name = '-'.join([
            'L' if args.lock_image_model else 'U',
            f'[{args.image_model.replace("/", "_")}-h{args.image_head_n_layers}]',
            'L' if args.lock_text_model else 'U',
            f'[{args.text_model.replace("/", "_")}-h{args.text_head_n_layers}]',
            f"b_{int(args.batch_size * args.world_size)}",
            f"ep_{args.epochs}",
            datetime.now().strftime("%m_%d-%H_%M_%S"),
        ])
    else:
        name = args.name

    name.replace('/', '_')
    if os.path.exists(os.path.join(args.logs, name)):
        time.sleep(1)
        name += '-'+datetime.now().strftime("%m_%d-%H_%M_%S")
        print(f"args.name is changed to '{name}' to avoid duplication.")
    return name