from glob import glob
import os
import shutil


def init_folders(dataset_path):
    print("Make sure you're not deleting something wrong")
    # try:
    #     shutil.rmtree(dataset_path+'darkfield')
    #     shutil.rmtree(dataset_path+'brightfield')
    #     shutil.rmtree(dataset_path+'fluorescent')
    # except:
    #     pass
    # os.makedirs(dataset_path+'darkfield')
    # os.makedirs(dataset_path+'brightfield')
    # os.makedirs(dataset_path+'fluorescent')
    # os.makedirs(dataset_path+'darkfield/clean')
    # os.makedirs(dataset_path+'brightfield/clean')
    # os.makedirs(dataset_path+'fluorescent/clean')
    # os.makedirs(dataset_path+'darkfield/tr4')
    # os.makedirs(dataset_path+'brightfield/tr4')
    # os.makedirs(dataset_path+'fluorescent/tr4')


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

def to_class_folder(dataset_path):
    josh = '../dataset/trash/josh/'
    images_path = glob(josh+'**/*.jpg', recursive=True)
    n = 0
    for path in images_path:
        type, label, new_name = path.split('/')[-3:]
        old_name = new_name.replace(" ", "")
        old_name = old_name[:-10] + " R" + str(int(old_name[-9:-7])) + old_name[-7:]

        # print(type)
        # print(label)
        # print(new_name)
        # print(old_name)
        print(dataset_path+type+'/'+old_name)
        print(dataset_path+type+'/'+label+'/'+new_name)
        try:
            os.rename(dataset_path+type+'/'+old_name, dataset_path+type+'/'+label+'/'+new_name)
        except:
            print(type+'/'+old_name)

        # if type == 'brightfield':
        #     n+=1
        #     print("#################")
        #     print(new_name)
        #     print(old_name)
        #     try:
        #         #move brightfield
        #         os.rename(dataset_path+'brightfield/'+old_name, dataset_path+'brightfield/'+label+'/'+new_name)
        #         #move darkfield
        #         new_name = new_name[:-6] + str(int(new_name[-6:-4]) + 1) + new_name[-4:]
        #         old_name = old_name[:-6] + str(int(old_name[-6:-4]) + 1) + old_name[-4:]
        #         os.rename(dataset_path+'darkfield/'+old_name, dataset_path+'darkfield/'+label+'/'+new_name)
        #         #move fluorescent
        #         new_name = new_name[:-6] + str(int(new_name[-6:-4]) + 1) + new_name[-4:]
        #         old_name = old_name[:-6] + str(int(old_name[-6:-4]) + 1) + old_name[-4:]
        #         os.rename(dataset_path+'fluorescent/'+old_name, dataset_path+'fluorescent/'+label+'/'+new_name)
        #     except:
        #         print(old_name+' not found')
    print(n)


if __name__ == '__main__':
    dataset_path = '../dataset/dataset_3/'
    #init_folders(dataset_path)
    # to_type_folder(dataset_path)
    to_class_folder(dataset_path)
