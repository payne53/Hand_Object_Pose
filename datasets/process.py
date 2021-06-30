import numpy as np
import os

old_root = '/S4/MI/zhuangn/hand_pose/datasets/obman/obman/'
new_root = '/ssd1/zhuangn/datasets/obman/obman/'

root = './ob'

train_data = np.load(os.path.join(root, 'images-train-icst.npy'))
for i in range(train_data.shape[0]):
    train_data[i] = train_data[i].replace(old_root, new_root)
np.save(os.path.join(root, 'images-train.npy'), train_data)

val_data = np.load(os.path.join(root, 'images-val-icst.npy'))
for i in range(val_data.shape[0]):
    val_data[i] = val_data[i].replace(old_root, new_root)
np.save(os.path.join(root, 'images-val.npy'), val_data)

test_data = np.load(os.path.join(root, 'images-test-icst.npy'))
for i in range(test_data.shape[0]):
    test_data[i] = test_data[i].replace(old_root, new_root)
np.save(os.path.join(root, 'images-test.npy'), test_data)
