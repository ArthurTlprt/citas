from glob import glob
import os
import shutil


def init_folders(dataset_path):
    # try:
    #     shutil.rmtree(dataset_path+'darkfield')
    #     shutil.rmtree(dataset_path+'brightfield')
    #     shutil.rmtree(dataset_path+'fluorescent')
    # except:
    #     pass
    os.makedirs(dataset_path+'darkfield')
    os.makedirs(dataset_path+'brightfield')
    os.makedirs(dataset_path+'fluorescent')

def get_info(file_path):
    file_name = file_path.split('/')[-1]
    row_number = file_path.split('/')[-2].split(' ')[-1]
    new_file_name = file_name[:5] + 'R' + row_number + file_name[5:]
    file_number = file_name[4:8]
    return int(file_number), file_name, new_file_name

def to_type_folder(dataset_path):
    images_path = glob(dataset_path+'**/*.jpg', recursive=True)
    for im_path in images_path:
        n, file_name, new_file_name = get_info(im_path)
        n %= 3
        if n == 0:
            # move to fluorescent
            os.rename(im_path, dataset_path+'fluorescent/'+new_file_name)
        elif n == 1:
            # move to brightfield
            os.rename(im_path, dataset_path+'brightfield/'+new_file_name)
        elif n == 2:
            # move to darkfield
            os.rename(im_path, dataset_path+'darkfield/'+new_file_name)

if __name__ == '__main__':
    dataset_path = '../dataset/dataset_3/'
    init_folders(dataset_path)
    to_type_folder(dataset_path)
