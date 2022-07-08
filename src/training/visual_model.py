import logging
from seed import models
from open_clip import create_model_and_transforms, create_transforms
from seed.utils import load_simclr_teacher_encoder, load_moco_teacher_encoder, load_swav_teacher_encoder


def get_visual_model_and_preprocess(args):

    # return a visual nn.module, where model.output_dim stores the feature dimension
    if args.open_clip_model:
        CLIP_model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            args=args
        )
        # CLIP model created by OpenCLIP has image and text tower,
        # remove text tower and leave the image tower as student.
        student = CLIP_model.visual
    else:
        # if 'moco' in args.pretrained:
        #     # FIXME: somehow the result of moco checkpoint is low
        #     student = models.__dict__[args.model](pretrained=False, num_classes=128)
        #     student = load_moco_teacher_encoder(args, student, logging, distributed=args.distributed)
        #     student.output_dim = 128
        # elif args.pretrained == 'simclr':
        #     # TODO
        #     student = load_simclr_teacher_encoder(args, student, logging, distributed=args.distributed)
        # elif args.pretrained == 'swav':
        #     # TODO
        #     student = load_swav_teacher_encoder(args, student, logging, distributed=args.distributed)
        # else:
        pretrained = (args.pretrained=='torchvision')
        logging.info(f'[torchvision]: loading {args.model} model, pretrained={pretrained}')
        student = models.__dict__[args.model](pretrained=pretrained, num_classes=1000)
        student.output_dim = 1000

        student.to(device=args.device)
        preprocess_train, preprocess_val = create_transforms(image_size=224, args=args)

    return student, preprocess_train, preprocess_val
