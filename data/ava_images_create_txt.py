# 获取数据集路径，创建数据集索引
import os
from tqdm import tqdm
import pandas as pd

# 获取数据集图像id
ava_image_path = '/home/flyingbird/Smith/Data/AVA/images'

ava_img_list = os.listdir(ava_image_path)
print(len(ava_img_list))

# for i in tqdm(ava_img_list):
#     if i[-1:-4] != 'jpg':
#         ava_img_list.remove(i)
ava_img_list.remove('final.log')
ava_img_list.remove('images')

print(len(ava_img_list))

# 将图像id和label对应
ava_label_path = 'AVA_with_segs_scores_aesthetic.txt'
df1 = pd.read_csv(ava_label_path, sep=' ')
label = []

ava_label_list = df1[['image_id', 'aesthetic_image']].tolist()

for i in ava_img_list:
    for j in ava_label_list:
        if i == j[0]:
            label.append(j[1])

# 构建数据集索引
ava_img_txt = 'AVA_img.txt'

with open(ava_img_txt, 'w') as f:
    for i in range(len(ava_img_list)):
        line = ava_image_path + '/' + ava_img_list[i] + '.jpg' + ' ' + label[i] + '\n'
        f.write(line)

# 使用pandas保存
#new_df_ava[['image_id', 'aesthetic_image']].to_csv('final_ava_list.txt', sep=' ', header=None, index=None)

