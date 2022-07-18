import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_wd(base_value, final_value, power, total_step):
    schedule = np.linspace(0, total_step, total_step, endpoint=False)**power / (total_step+1)**power
    schedule = schedule * (final_value-base_value) + base_value
    
    assert len(schedule) == total_step
    return schedule

# for power in [1,2,4,8,16,32]: 
#     wd_scheduler = cosine_wd(base_value=0.1, final_value=10, epochs=64, niter_per_ep=1000, power=power)

#     print(wd_scheduler) 
#     import matplotlib.pyplot as plt
#     plt.plot(wd_scheduler, label=f'power={power}')

# plt.legend()
# plt.savefig('wd_curve.png')