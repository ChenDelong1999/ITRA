from torch import nn
from utils.get_model_from_log_dir import get_model_from_log_dir


# class Example(nn.Module):
#     def forward_features():
#         """
#         This method is called to extract features for evaluation.
#         """
#         pass


# def get_cls_model(config, **kwargs):
#     """
#     Specify your model here
#     """
#     model = Example()
#     return model

def get_cls_model(config, **kwargs):
    """
    Specify your model here
    """
    # model = Example()
    model = get_model_from_log_dir(exp_dir=config.log_dir, epoch=config.ckpt_epoch)
    model.forward_features = model.encode_image
    return model
