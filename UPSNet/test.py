from upsnet.dataset.json_gta5_dataset import JsonGTA5Dataset

import logging

logger = logging
logging.basicConfig(level=logging.DEBUG)

a = JsonGTA5Dataset('gta5', '/mnt/data/gta5/train', '/mnt/data/gta5/train/inst.json')
a.get_roidb(gt=True)
