
from training.model import get_model
import copy

def get_wise_ft_model(finetuned_model, args, alpha=0.5):
    finetuned_model_without_ddp = finetuned_model.module if args.distributed else finetuned_model
    args_ = copy.deepcopy(args)
    args_.distributed = False

    zeroshot_model, preprocess_train, preprocess_val, preprocess_aug = get_model(args_)
    theta_0 = zeroshot_model.state_dict()
    theta_1 = finetuned_model_without_ddp.state_dict()

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }

    # update the model acccording to the new weights
    zeroshot_model.load_state_dict(theta)

    return zeroshot_model