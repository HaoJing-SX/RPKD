import numpy as np
import os

train_txtdir = '/home/work/waymo/ImageSets/train.txt'
val_txtdir = '/home/work/waymo/ImageSets/val.txt'
npy_folderdir = '/home/work/waymo/waymo_processed_data_v0_5_0/'
out_folderdir = '/home/work/waymo/waymo_bin_20sq_10val/'
out_traintxt_dir = '/home/work/waymo/waymo_bin_20sq_10val/waymo30training.txt'
out_testtxt_dir = '/home/work/waymo/waymo_bin_20sq_10val/waymo30testing.txt'

train_sample_sequence_list = [x.strip() for x in open(train_txtdir).readlines()]
val_sample_sequence_list = [x.strip() for x in open(val_txtdir).readlines()]
sequence_name = []
for k in range(len(train_sample_sequence_list)):
    sequence_name.append(os.path.splitext(train_sample_sequence_list[k])[0])
for k in range(len(val_sample_sequence_list)):
    sequence_name.append(os.path.splitext(val_sample_sequence_list[k])[0])
# search npy
train_txt = []
test_txt = []
for i in range(len(sequence_name)):
    forder_dir = npy_folderdir + sequence_name[i]
    # files = [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(forder_dir) for file in files]
    files = [os.path.join(forder_dir, f) for f in os.listdir(forder_dir)]
    files.sort()
    if i <= 9:
        folder_indices = "0" + str(i)
    elif i > 9 and i <= 99:
        folder_indices = str(i)
    out_sub_folderdir = out_folderdir + folder_indices + "/velodyne"
    if not os.path.exists(out_sub_folderdir):
        os.makedirs(out_sub_folderdir)
    for file in files:
        if file.endswith(".npy"):
            file_name = ("00" + os.path.basename(file)).rstrip(".npy")
            out_dir = out_sub_folderdir + "/" + file_name + ".bin"
            # data = np.load(file)
            # data.tofile(out_dir)
            # relative path to txt
            rpath = folder_indices + "/velodyne/" + file_name + ".bin"
            if i <= 19:
                train_txt.append(rpath)
            else:
                test_txt.append(rpath)
    print("Convert_" + folder_indices + "_Over")

with open(out_traintxt_dir, "w") as file:
    for line in train_txt:
        file.write(line + "\n")
with open(out_testtxt_dir, "w") as file:
    for line in test_txt:
        file.write(line + "\n")
print("TXT_Over")
