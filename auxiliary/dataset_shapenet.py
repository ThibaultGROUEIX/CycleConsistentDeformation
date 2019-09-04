# TODO : remove `fine_tune` if useless
# TODO : rename file to dataset_shapenet.py

from __future__ import print_function
import torch.utils.data as data
import os.path
import json
import random
from auxiliary.meter import *
from joblib import Parallel, delayed
from collections import defaultdict
import joblib
import pickle
from sklearn.neighbors import NearestNeighbors
import my_utils
import time
import normalize_points
import torch
import useful_losses as loss


def unwrap_self(arg, **kwarg):
    return arg[0]._getitem(*(arg[1:]), **kwarg)


class ShapeNetSeg(data.Dataset):
    def __init__(self, root="./data/dataset_shapenet", num_samples=-1, normalization="BoundingBox", knn=False,
                 num_neighbors=20,
                 class_choice=None, mode="TRAIN", normal=False, npoints=2500, sample=True,
                 data_augmentation_Z_rotation=False, data_augmentation_Z_rotation_range=360,
                 data_augmentation_3D_rotation=False, random_translation=False, anisotropic_scaling=False,
                 shuffle=False, get_single_shape=False):
        self.dataset_string_args = str(mode) + "_" + \
                                   str(class_choice) + \
                                   "_" + str(num_samples) + \
                                   "_" + str(normalization) + \
                                   "_" + str(knn) + \
                                   "_" + str(num_neighbors) + \
                                   "_" + str(shuffle) + \
                                   "_" + str(normal) + \
                                   "_" + str(npoints) + \
                                   "_" + str(sample) + \
                                   "_" + str(data_augmentation_Z_rotation) + \
                                   "_" + str(data_augmentation_Z_rotation_range) + \
                                   "_" + str(data_augmentation_3D_rotation) + \
                                   "_" + str(random_translation) + \
                                   "_" + str(anisotropic_scaling)
        self.path_dataset = os.path.join("./data/", self.dataset_string_args)
        self.shuffle = shuffle
        self.num_samples = num_samples
        self.anisotropic_scaling = anisotropic_scaling
        self.fine_tune = (mode == 'Fine_tune_test')
        if mode == 'Fine_tune_test':
            mode = 'ALLDATA'
        self.knn = knn
        self.num_neighbors = num_neighbors
        self.normalization = normalization
        self.random_translation = random_translation
        self.data_augmentation_Z_rotation = data_augmentation_Z_rotation
        self.data_augmentation_Z_rotation_range = data_augmentation_Z_rotation_range  # range in degree of random rotation
        self.data_augmentation_3D_rotation = data_augmentation_3D_rotation
        self.npoints = npoints
        self.sample = sample
        self.normal = normal
        self.mode = mode
        self.root = root
        self.get_single_shape = get_single_shape
        self.datapath = []  # List to store all path of dataset files
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.namecat2numbercat = {}
        self.numbercat2namecat = {}
        self.numsamples_by_cat = {}
        self.class_choice = class_choice
        self.meta = {}
        self.data = []

        if not os.path.exists(self.root):
            print("Downloading Shapetnet for segmentation...")
            os.system('chmod +x ./data/download_dataset_shapenet.sh')
            os.system('./data/download_dataset_shapenet.sh')
        # ----------------------------------------------------------#
        ## Create dictionaries with keys : name of class, value : name of folder and vice-versa
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.numsamples_by_cat[ls[0]] = 0
                self.namecat2numbercat[ls[0]] = ls[1]
                self.numbercat2namecat[ls[1]] = ls[0]
                self.meta[ls[0]] = []  # List to store all path of files per category

        # ----------------------------------------------------------#
        # create list of files datapaths
        if self.mode == "TRAIN":
            with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_train_file_list.json')) as f:
                file_list = json.load(f)
        if self.mode == "VAL":
            with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_val_file_list.json')) as f:
                file_list = json.load(f)
        if self.mode == "TEST":
            with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_test_file_list.json')) as f:
                file_list = json.load(f)
        if self.mode == "ALLDATA":
            file_list = []
            with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_train_file_list.json')) as f:
                file_list = file_list + json.load(f)
                self.len_train = len(file_list)
            with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_val_file_list.json')) as f:
                file_list = file_list + json.load(f)
                self.len_val = len(file_list) - self.len_train
            with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_test_file_list.json')) as f:
                file_list = file_list + json.load(f)
                self.len_test = len(file_list) - self.len_train - self.len_val

        for file in file_list:
            # Typical example : shape_data/03001627/355fa0f35b61fdd7aa74a6b5ee13e775 so remove 'shape_data/' and add '.txt'
            file_path = os.path.join(self.root, file[11:]) + ".txt"
            number_category = file[11:19] + ""
            if (class_choice is None) or (self.numbercat2namecat[number_category] in class_choice):
                self.meta[self.numbercat2namecat[number_category]].append(
                    (file_path, self.numbercat2namecat[number_category]))
                self.numsamples_by_cat[self.numbercat2namecat[number_category]] = self.numsamples_by_cat[
                                                                                      self.numbercat2namecat[
                                                                                          number_category]] + 1

        # ----------------------------------------------------------#
        # add all paths to the same list, keep track of sizes
        for cat in self.meta.keys():
            for file in self.meta[cat]:
                self.datapath.append(file)

        # ----------------------------------------------------------#
        # Compute number of segments
        self.num_segment = 50  # from PointNet++ paper
        # ----------------------------------------------------------#

        # get 1 meter per category
        self.perCatValueMeter = {}
        for cat in self.meta.keys():
            self.perCatValueMeter[cat] = AverageValueMeter()
        # ----------------------------------------------------------#
        try:
            with open(os.path.join(self.root, 'parts_by_cat_' + str(class_choice) + '.json'), 'r') as fp:
                self.part_category = json.load(fp)
            print("Reload parts from : ", os.path.join(self.root, 'parts_by_cat_' + str(class_choice) + '.json'))

        except:
            self.part_category = self.generate_parts_by_cat()
            with open(os.path.join(self.root, 'parts_by_cat_' + str(class_choice) + '.json'), 'w') as fp:
                json.dump(self.part_category, fp)
                # LOGS
        my_utils.red_print("size of " + self.mode + " dataset : " + str(len(self.datapath)))

        my_utils.red_print("Dataset normalization : " + self.normalization)
        if self.normalization == "UnitBall":
            self.normalization_function = normalize_points.UnitBall
        elif self.normalization == "BoundingBox":
            self.normalization_function = normalize_points.BoundingBox
        elif self.normalization == "BoundingBox_2":
            self.normalization_function = normalize_points.BoundingBox_2
        else:
            self.normalization_function = normalize_points.identity

        self.preprocess()
        if self.knn:
            start = time.time()
            my_utils.red_print(
                "Computing nearest neighbors graph... (can take some time if it's not already precomputed.)")
            self.compute_nearest_neighbors_graph()
            my_utils.red_print('Done!')
            end = time.time()
            my_utils.red_print("Ellapsed time : " + '"%.2f' % (end - start))

        # Shuffle_list
        self.len_data = self.__len__()
        self.shuffle_list = [i for i in range(self.len_data)]
        if self.shuffle:
            random.shuffle(self.shuffle_list)
            random.shuffle(self.shuffle_list)

    def compute_nearest_neighbors_graph(self):
        if not os.path.exists(self.path_dataset + "_knn_indices.npy"):
            knn = [data[0] for data in self.datas]
            knn = my_utils.uniformize_sizes(knn)
            self.num_neighbors = min(self.num_neighbors, len(self.datas))
            nbrs = NearestNeighbors(n_neighbors=self.num_neighbors, algorithm='ball_tree', metric=loss.NN_metric,
                                    n_jobs=4).fit(knn.view(knn.size(0), -1).numpy())
            distances, indices = nbrs.kneighbors(knn.view(knn.size(0), -1).numpy())
            self.indices = indices
            self.distances = distances
            np.save(self.path_dataset + "_knn_indices.npy", self.indices)
            np.save(self.path_dataset + "_knn_distances.npy", self.distances)
        else:
            self.indices = np.load(self.path_dataset + "_knn_indices.npy")
            self.distances = np.load(self.path_dataset + "_knn_distances.npy")

    def _getitem(self, index):
        file = self.datapath[index]
        points = np.loadtxt(file[0])
        points = torch.from_numpy(points).float()
        # Normalization is done before resampling !
        points[:, :3] = self.normalization_function(points[:, :3])
        return points, file[1], file[0]

    def preprocess(self):
        start = time.time()
        if os.path.exists(self.path_dataset + ".pkl"):
            print("Reload dataset : ", self.path_dataset)
            with open(self.path_dataset + ".pkl", "rb") as fp:
                self.datas = pickle.load(fp)
        else:
            my_utils.red_print("preprocess dataset...")

            class BatchCompletionCallBack(object):
                completed = defaultdict(int)

                def __init__(se, time, index, parallel):
                    se.index = index
                    se.parallel = parallel

                def __call__(se, index):
                    BatchCompletionCallBack.completed[se.parallel] += 1
                    if BatchCompletionCallBack.completed[se.parallel] % 100 == 0:
                        end = time.time()
                        etl = (end - start) * (
                                self.__len__() / float(BatchCompletionCallBack.completed[se.parallel])) - (
                                      end - start)
                        print('\r' + "Progress : %f %% " %
                              float(BatchCompletionCallBack.completed[
                                        se.parallel] * 100 / self.__len__()) + "ETL %d seconds" % int(etl), end='')
                    if se.parallel._original_iterator is not None:
                        se.parallel.dispatch_next()

            joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack
            self.datas = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(unwrap_self)(i) for i in zip([self] * self.__len__(), range(self.__len__())))

            with open(self.path_dataset + ".pkl", "wb") as fp:  # Pickling
                pickle.dump(self.datas, fp)

        my_utils.red_print(" dataset : " + str(len(self.datas)))
        end = time.time()
        my_utils.red_print("Ellapsed time : " + '"%.2f' % (end - start))

    def generate_parts_by_cat(self):
        parts_by_cat = {}
        for cat in self.meta.keys():
            labels = set([])
            for file in self.meta[cat]:
                points = np.loadtxt(file[0])
                labels = labels | set(points[:, 6])
            print(cat, labels)
            parts_by_cat[cat] = list(labels)
        return parts_by_cat

    def getAnItem(self, index):
        # ----------------------------------------------------------#
        points, cat, file_path = self.datas[index]
        points = points.clone()
        # Resample
        if self.sample:
            choice = np.random.choice(points.size(0), self.npoints, replace=True)
            points = points[choice, :]

        rot_matrix = normalize_points.uniform_rotation_axis_matrix(axis=1,
                                                                   range_rot=self.data_augmentation_Z_rotation_range)
        if self.data_augmentation_Z_rotation:
            # Uniform random Rotation of axis Y
            points, rot_matrix = normalize_points.uniform_rotation_axis(points, axis=1, normals=self.normal,
                                                                        range_rot=self.data_augmentation_Z_rotation_range)
        if self.anisotropic_scaling:
            # Data augmentation : anisotropic scaling
            points[:, :3] = normalize_points.anisotropic_scaling(points[:, :3]).contiguous()
            points[:, :3] = self.normalization_function(points[:, :3])

        if self.data_augmentation_3D_rotation:
            #  Uniform random 3D rotation of the sphere.
            points, rot_matrix = normalize_points.uniform_rotation_sphere(points, normals=self.normal)
        # Remark : if input of data_augmentation is normalized to unit ball and centered, the output rotated is as well.

        if self.random_translation:
            points = normalize_points.add_random_translation(points, scale=0.03)
        return points, cat, rot_matrix, file_path
        # ----------------------------------------------------------#

    def __getitem__(self, index):
        index = self.shuffle_list[index]
        if self.fine_tune:
            index = index + self.len_train + self.len_val
        if self.knn:
            index_2 = np.random.randint(self.num_neighbors)
            index_3 = np.random.randint(self.num_neighbors)
            return self.getAnItem(index) + self.getAnItem(self.indices[index][index_2]) + self.getAnItem(
                    self.indices[index][index_3])
        else:
            if self.get_single_shape:
                return self.getAnItem(index)
            else:
                index_2 = np.random.randint(self.__len__())
                index_3 = np.random.randint(self.__len__())
                return self.getAnItem(index) + self.getAnItem(index_2) + self.getAnItem(index_3)

    def __len__(self):
        if self.num_samples > 0:
            return self.num_samples
        else:
            if self.fine_tune:
                return self.len_test
            else:
                return len(self.datapath)


if __name__ == '__main__':
    print('Testing Shapenet dataloader')

    d = ShapeNetSeg(mode="TEST", knn=False, sample=False, class_choice="Chair",
                    data_augmentation_Z_rotation=True, data_augmentation_Z_rotation_range=40, npoints=400,
                    random_translation=True, anisotropic_scaling=True, shuffle=True)
    print(d.shuffle_list)

    # d = ShapeNetSeg(mode="TEST", knn=False, sample=False, class_choice="Chair",
    #                 data_augmentation_Z_rotation=True, data_augmentation_Z_rotation_range=40, npoints=400,
    #                 random_translation=True, anisotropic_scaling=True)
    # # sys.path.append('/home/thibault/projects/workflow_and_installs/')
    # import workflow.save_mesh_from_points_and_labels as save_mesh_from_points_and_labels
    # for i in range(10):
    #     print(i)
    #     save_mesh_from_points_and_labels.save_mesh_from_pointsandlabels(d[0][0][:, :3], path=str(i)+"bb3.ply")
    #
    # d = ShapeNetSeg(mode="TEST", knn=False, sample=False, class_choice="Chair",
    #                 data_augmentation_Z_rotation=False, data_augmentation_Z_rotation_range=40, npoints=400,
    #                 random_translation=False, anisotropic_scaling=False)
    # a = d[0]
    # # sys.path.append('/home/thibault/projects/workflow_and_installs/')
    # import workflow.save_mesh_from_points_and_labels as save_mesh_from_points_and_labels
    # save_mesh_from_points_and_labels.save_mesh_from_pointsandlabels(a[0][:, :3], path="bb3_noscale.ply")

    # d  =  ShapeNetSeg(mode="TRAIN", normalization="BoundingBox",sample = False, class_choice="Cap", data_augmentation_Z_rotation=False, data_augmentation_Z_rotation_range = 40, npoints = 400, random_translation = False)
    # a = d[0]
    # save_mesh_from_points_and_labels.save_mesh_from_pointsandlabels(a[0][:, :3], path="bb2.ply")

    # d  =  ShapeNetSeg(mode="TRAIN", normalization="BoundingBox_2",sample = False, class_choice="Cap", data_augmentation_Z_rotation=False, data_augmentation_Z_rotation_range = 40, npoints = 400, random_translation = False)
    # a = d[0]
    # save_mesh_from_points_and_labels.save_mesh_from_pointsandlabels(a[0][:, :3], path="bb1.ply")
