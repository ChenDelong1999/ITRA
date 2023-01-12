from torch import nn
from utils.get_model_from_log_dir import get_model_from_log_dir

class Example(nn.Module):
    def encode_image():
        """
        This method is called to extract image features for evaluation.
        """
        pass

    def encode_text():
        """
        This method is called to extract text features for evaluation.
        """
        pass


def get_zeroshot_model(config, **kwargs):
    """
    Specify your model here
    """
    # model = Example()
    model = get_model_from_log_dir(exp_dir=config.log_dir, epoch=config.ckpt_epoch)
    return model
