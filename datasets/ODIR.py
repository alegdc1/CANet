import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from PIL import Image
import glob
import xlrd
import numpy as np
import scipy.misc


class traindataset(data.Dataset):
    def __init__(self, root, mode, transform=None, num_class=8, multitask=False, args=None):
        self.root = "./data/ODIR/"
        self.transform = transform
        self.mode = mode
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.name = []
        self.num_class = num_class
        self.multitask = multitask
        self.ignored_images = set()

        # get file path
        # files = glob.glob(self.root+'/*/*.tif')
        # self.train_root  = files
        # get file name and label
        xls_files = glob.glob(self.root + '/*/*.xls')
        dictLabels = self.load_csv(xls_files)

        # todo generate file_list of all left/right images in dataset
        files = np.loadtxt(self.root + "file_list.txt", dtype=str)
        # only take first 100 train images
        files = files[0:100]

        # todo generate the folds
        # idx = np.loadtxt(self.root + "/10fold/" + str(args.fold_name) + ".txt", dtype=int)

        # test only with 3 images for now
        #idx = np.random.choice(range(7000), 3, replace=False)
        idx = [0,1,2]

        self.test_root = [files[idx_item] for idx_item in idx]
        self.train_root = list(set(files) - set(self.test_root))
        self.train_root = [self.root + item for item in self.train_root]
        self.test_root = [self.root + item for item in self.test_root]

        # self.train_root = np.genfromtxt("train_root.txt", dtype=str)
        # self.test_root = np.genfromtxt("test_root.txt", dtype=str)
        # np.savetxt("train_root.txt", self.train_root, fmt="%s")
        # np.savetxt("test_root.txt", self.test_root, fmt="%s")
        # print ("test sample", self.test_root[0])

        if self.mode == 'train':
            # self.train_root = [item.replace("/medical/", "/Documents/datasets/medical/") for item in self.train_root]
            for each_one in self.train_root:
                img_file = each_one.split("/")[-1]

                # skip bad images, like "no fundus image" or "low img quality"
                if each_one in self.ignored_images:
                    continue

                # create network input (train data, label)
                # dictLabels = {"N" : ['/data/1_right.tif', '']}
                labels = [k for k, v in dictLabels.items() if img_file in v]

                # labels = {"G", "C", "A"}
                label_vector = self.to_one_hot_vector(labels)

                self.train_label.append(label_vector)

                self.train_data.append(img_file)
                self.name.append(img_file)
            assert len(self.train_label) == len(self.train_data)

            print('=> Total Train: ', len(self.train_data), " ODIR images ")

        elif self.mode == 'val':
            # print(self.test_root)
            # self.test_root = [item.replace("/medical/", "/Documents/datasets/medical/") for item in self.test_root]

            for item in self.test_root:
                img_file = item.split("/")[-1]

                if item in self.ignored_images:
                    continue

                # dictLabels = {"N" : ['/data/1_right.tif', '']}
                labels = [k for k, v in dictLabels.items() if img_file in v]

                # labels = {"G", "C", "A"}
                label_vector = self.to_one_hot_vector(labels)

                self.test_label.append(label_vector)

                #TODO: instead of appending image, append file name
                self.test_data.append(img_file)
                self.name.append(img_file)
            assert len(self.test_data) == len(self.test_label)

            print('=> Total Test: ', len(self.test_data), " ODIR images ")

    def to_one_hot_vector(self, label_set):
        label_vector = np.zeros(8)

        if "N" in label_set:
            label_vector[0] = 1
        if "D" in label_set:
            label_vector[1] = 1
        if "G" in label_set:
            label_vector[2] = 1
        if "C" in label_set:
            label_vector[3] = 1
        if "A" in label_set:
            label_vector[4] = 1
        if "H" in label_set:
            label_vector[5] = 1
        if "M" in label_set:
            label_vector[6] = 1
        if "O" in label_set:
            label_vector[7] = 1

        return label_vector

    @staticmethod
    def extract_labels(input_label):
        inputs = input_label.split(",")

        ignored_labels = ["lens dust"]
        ignored_keywords = ["low image quality", "no fundus image"]
        ignore_image = False

        labels = set()

        for split in inputs:
            if split.strip() in ignored_keywords:
                ignore_image = True
                break

            if split.strip() in ignored_labels:
                print("Ignoring ", split)
                continue

            has_found = False
            if "cataract" in split:
                labels.add("C")
                has_found = True
            if "myopia" in split:
                labels.add("M")
                has_found = True
            if "hypertensive retinopathy" in split:
                labels.add("H")
                has_found = True
            if "age-related" in split:
                labels.add("A")
                has_found = True
            if "glaucoma" in split:
                labels.add("G")
                has_found = True
            if "proliferative retinopathy" in split:
                labels.add("D")
                has_found = True

            # only normal fundus if others are not set
            if "normal fundus" in split and not has_found:
                labels.add("N")
                has_found = True

            # check for others
            if not has_found:
                labels.add("O")

        label_names = ["D", "G", "A", "C", "H", "M", "O"]
        flag = False
        for lname in label_names:
            if lname in labels:
                flag = True
                break

        if flag and ("N" in labels):
            labels.remove("N")

        # if ("D" in labels) or ("G" in labels) or ("C" in labels) or ("A" in labels) or ("H" in labels) or ("M" in labels) or ("O" in labels) and ("N" in labels):
        #   labels.remove("N")

        return ignore_image, labels

    def load_csv(self, path):

        dictLabels = {}
        for per_path in path:
            # open xlsx
            xl_workbook = xlrd.open_workbook(per_path)
            xl_sheet = xl_workbook.sheet_by_index(0)
            for rowx in range(1, xl_sheet.nrows):
                cols = xl_sheet.row_values(rowx)
                filename_l = cols[3]
                filename_r = cols[4]
                label_l = int(cols[5])
                label_r = int(cols[6])

                ignore_r, labels_r = self.extract_labels(label_r)
                ignore_l, labels_l = self.extract_labels(label_l)

                if not ignore_r:
                    for label in labels_r:
                        dictLabels[label].append(filename_r)
                else:
                    self.ignored_images.add(filename_r)

                if not ignore_l:
                    for label in labels_l:
                        dictLabels[label].append(filename_l)
                else:
                    self.ignored_images.add(filename_r)

        # print (len(dictLabels_DR[0])) 546
        # print (len(dictLabels_DR[1])) 153
        # print (len(dictLabels_DR[2])) 247
        # print (len(dictLabels_DR[3])) 254
        #
        # print (len(dictLabels_DME[0])) 974
        # print (len(dictLabels_DME[1])) 75
        # print (len(dictLabels_DME[2])) 151
        return dictLabels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.mode == 'train':
            img_file, label, name = self.train_data[index], self.train_label[index], self.name[index]
        elif self.mode == 'val':
            img_file, label, name = self.test_data[index], self.test_label[index], self.name[index]

        img = Image.open(img_file)
        img = img.convert('RGB')

        img = self.transform(img)

        return img, label, name

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.test_data)
