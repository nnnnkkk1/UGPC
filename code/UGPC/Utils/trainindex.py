import os
from sklearn.model_selection import train_test_split

# 预处理输出地址
data_path = "/mnt/UPCPRO/Training_Set"#Training_Set地址
train_and_test_ids = os.listdir(data_path)

train_ids, test_ids = train_test_split(train_and_test_ids, test_size=0.2,random_state=6)
#train_ids, val_ids = train_test_split(train_ids, test_size=0,random_state=6)
print("Using {} images for training,  {} images for testing.".format(len(train_ids),len(test_ids)))

with open('/mnt/UPCPRO/train.list','w') as f:
    f.write('\n'.join(train_ids))#trainlist保存地址


with open('/mnt/UPCPRO/test.list','w') as f:
    f.write('\n'.join(test_ids))#testlist保存地址

# with open('/mnt/UPCPRO/val.list','w') as f:
#     f.write('\n'.join(val_ids))#vallist保存地址
