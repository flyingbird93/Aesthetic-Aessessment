filepath = '100000_ava_list.txt'
savepath = '100000_img_path.txt'
img_list = []

with open(filepath, 'r') as f:
    for line in f.readlines():
        img_list.append(line.split(' ')[0])


with open(savepath, 'w') as f:
    for i in img_list:
        line = '/home/flyingbird/Smith/Data/AVA/images/' + i + '.jpg' + '\n'
        f.write(line)