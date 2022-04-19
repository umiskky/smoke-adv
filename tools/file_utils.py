import os
import os.path as osp
from datetime import datetime, timezone, timedelta

import torch


def get_utc8_time():
    UTF8_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(UTF8_TZ)
    return beijing_now.strftime('%Y-%m-%d-%H-%M')


def makedirs(base: str, *target):
    if len(target) == 0:
        if not osp.exists(base):
            os.makedirs(base)
    else:
        if not osp.exists(osp.join(base, *target)):
            os.makedirs(osp.join(base, *target))


def state_saving(state: dict, epoch: int, loss: float, path: str):
    saving_path = osp.join(path, "adv_texture")
    makedirs(saving_path)
    file_name = "%05d" % epoch + "_" + ("%.5f" % abs(loss)).zfill(9) + "_adv_texture_hls.pth"
    torch.save(state, osp.join(saving_path, file_name))


def update_dic(dic, base_dic):
    """Update config from dic based base_dic"""
    base_dic = base_dic.copy()
    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            base_dic[key] = update_dic(val, base_dic[key])
        else:
            base_dic[key] = val
    dic = base_dic
    return dic
