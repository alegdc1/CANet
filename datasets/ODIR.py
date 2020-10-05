import sys
import pdb

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from PIL import Image
import glob
import xlrd
import numpy as np


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
        self.dataset = []
        self.multitask = multitask
        self.ignored_images = set()

        # get file path
        # files = glob.glob(self.root+'/*/*.tif')
        # self.train_root  = files
        # get file name and label
        xls_files = glob.glob(self.root + '*.xlsx')
        dictLabels = self.load_csv(xls_files)
        # generate file_list of all left/right images in dataset
        files = np.loadtxt(self.root + "file_list.txt", dtype=str)
        # choose how many samples you want, starts at first one
        files = files[0:7000]

        # generate the folds
        # idx = np.loadtxt(self.root + "/10fold/" + str(args.fold_name) + ".txt", dtype=int)

        idx = np.loadtxt("/scratch_net/petzi/garciaal/git/CANet/data/ODIR/valid_idces.txt", dtype='int')

        self.test_root = [files[idx_item] for idx_item in idx]
        self.train_root = list(set(files) - set(self.test_root))
        self.train_root = [self.root + item for item in self.train_root]
        self.test_root = [self.root + item for item in self.test_root]

        # self.train_root = np.genfromtxt("train_root.txt", dtype=str)
        # self.test_root = np.genfromtxt("test_root.txt", dtype=str)
        # np.savetxt("train_root.txt", self.train_root, fmt="%s")
        # np.savetxt("test_root.txt", self.test_root, fmt="%s")
        # print ("test sample", self.test_root[0])
        sum_vector = np.zeros(8)
        counter = 0
        if self.mode == 'train':
            # self.train_root = [item.replace("/medical/", "/Documents/datasets/medical/") for item in self.train_root]
            for each_one in self.train_root:
                img_file = each_one.split("/")[-1]

                # skip bad images, like "no fundus image" or "low img quality"
                if img_file in self.ignored_images:
                    continue
                # create network input (train data, label)
                # dictLabels = {"N" : ['/data/1_right.tif', '']}
                labels = [k for k, v in dictLabels.items() if img_file in v]

                # pdb.set_trace()
                # labels = {"G", "C", "A"}
                label_vector = self.to_one_hot_vector(labels)

                sum_vector += label_vector
                counter += not np.any(label_vector)
                self.train_label.append(label_vector)
                self.train_data.append(each_one)
                self.name.append(img_file)
                self.dataset.append('ODIR')
            assert len(self.train_label) == len(self.train_data)

            print('=> Total Train: ', len(self.train_data), " ODIR images ")
            print("sum_vector", sum_vector)
            print("counter", counter)

            # pdb.set_trace()

        elif self.mode == 'val':
            # print(self.test_root)
            # self.test_root = [item.replace("/medical/", "/Documents/datasets/medical/") for item in self.test_root]

            for item in self.test_root:
                img_file = item.split("/")[-1]

                if img_file in self.ignored_images:
                    continue

                # dictLabels = {"N" : ['/data/1_right.tif', '']}
                labels = [k for k, v in dictLabels.items() if img_file in v]

                # labels = {"G", "C", "A"}
                label_vector = self.to_one_hot_vector(labels)
                self.test_label.append(label_vector)

                self.test_data.append(item)
                self.name.append(img_file)
            assert len(self.test_data) == len(self.test_label)

            print('=> Total Test: ', len(self.test_data), " ODIR images ")

        # total = np.sum(sum_vector)
        # for i in range(8):
        #     print((1.0 / sum_vector[i]) * total / 2)

    def to_one_hot_vector(self, label_set):
        label_vector = np.zeros(8, dtype=np.double)

        if "N" in label_set:
            label_vector[0] = 1.
        if "D" in label_set:
            label_vector[1] = 1.
        if "G" in label_set:
            label_vector[2] = 1.
        if "C" in label_set:
            label_vector[3] = 1.
        if "A" in label_set:
            label_vector[4] = 1.
        if "H" in label_set:
            label_vector[5] = 1.
        if "M" in label_set:
            label_vector[6] = 1.
        if "O" in label_set:
            label_vector[7] = 1.
        # pdb.set_trace()
        return label_vector

    @staticmethod
    def extract_labels(input_label):
        # pdb.set_trace()
        inputs = input_label.split(",")

        ignored_labels = ["lens dust"]
        ignored_keywords = ["low image quality", "no fundus image"]
        ignore_image = False

        labels = set()

        for split in inputs:
            # pdb.set_trace()
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
            if "myopia" in split or "myopic retinopathy" in split or "myopic maculopathy" in split:
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
            if "proliferative retinopathy" in split or "diabetic retinopathy" in split:
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
        # dictLabels = dictLabels.get(labels)
        for per_path in path:
            # open xlsx
            xl_workbook = xlrd.open_workbook(per_path)
            xl_sheet = xl_workbook.sheet_by_index(0)
            for rowx in range(1, xl_sheet.nrows):
                cols = xl_sheet.row_values(rowx)
                filename_l = cols[3]
                filename_r = cols[4]
                label_l = cols[5]
                label_r = cols[6]

                ignore_r, labels_r = self.extract_labels(label_r)
                ignore_l, labels_l = self.extract_labels(label_l)
                # dictLabels = dictLabels.get(label_r, label_l)

                # pdb.set_trace()
                if not ignore_r:
                    # pdb.set_trace()
                    for label in labels_r:
                        # pdb.set_trace()
                        if label in dictLabels.keys():
                            dictLabels[label].append(filename_r)
                        else:
                            dictLabels[label] = [filename_r]
                        # pdb.set_trace()
                else:
                    self.ignored_images.add(filename_r)

                if not ignore_l:
                    for label in labels_l:
                        if label in dictLabels.keys():
                            dictLabels[label].append(filename_l)
                        else:
                            dictLabels[label] = [filename_l]
                else:
                    self.ignored_images.add(filename_l)

        return dictLabels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        ds = "ODIR"
        if self.mode == 'train':
            img_file, label, name, ds = self.train_data[index], self.train_label[index], self.name[index], self.dataset[index]
        elif self.mode == 'val':
            img_file, label, name = self.test_data[index], self.test_label[index], self.name[index]
        img = Image.open(img_file)
        img = img.convert('RGB')

        img = self.transform(img)


        return img, label, name, ds

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.test_data)
