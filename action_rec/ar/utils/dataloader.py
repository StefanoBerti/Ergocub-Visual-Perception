import copy
import os
import pickle
import torch.utils.data as data
import random
import numpy as np
import cv2
from action_rec.ar.utils.configuration import ubuntu


# https://rose1.ntu.edu.sg/dataset/actionRecognition/


def rotate_y(points, angle):  # Created by Chat GPT
    """Rotate an array of 3D points around the y-axis by a given angle (in degrees).

  Args:
    points: An Nx3 array of 3D points.
    angle: The angle (in degrees) to rotate the points around the y-axis.

  Returns:
    An Nx3 array of rotated points.
  """
    # Convert the angle to radians
    angle = np.radians(angle)

    # Create the rotation matrix
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])

    # Rotate the points
    rotated_points = np.dot(points, R.T)

    return rotated_points


class MyLoader(data.Dataset):
    def __init__(self, queries_path, k=5, n=5, n_task=10000, max_l=16, l=8, input_type="hybrid",
                 exemplars_path=None, support_classes=None, query_class=None,
                 skeleton="smpl+head_30", given_122=False, do_augmentation=True):
        """
        Loader class that provides all the functionality needed for training and testing
        @param queries_path: path to main dataset
        @param k: dimension of support set (ways)
        @param n: number of element inside the support set for each class (shots)
        @param n_task: number of task for each epoch, if query_class is not provided
        @param max_l: expected maximum number of frame for each instance
        @param l: number of frame to load for each instance
        @param input_type: one between ["skeleton", "rgb", "hybrid"]
        @param exemplars_path: if provided, support set elements will be loaded from this folder
        @param support_classes: if provided, the support set will always contain these classes
        @param query_class: if provided, queries will belong only from this class
        """
        self.queries_path = queries_path
        self.k = k
        self.n = n
        self.max_l = max_l
        self.l = l
        self.input_type = input_type
        self.all_classes = next(os.walk(self.queries_path))[1]  # Get list of directories
        self.do_augmentation = do_augmentation

        self.support_classes = support_classes  # Optional, to load always same classes in support set
        self.exemplars_path = exemplars_path  # Optional, to use exemplars when loading support set

        self.n_task = n_task
        self.query_class = query_class
        self.queries = None
        if self.query_class:
            self.queries = []
            for class_dir in next(os.walk(os.path.join(queries_path, query_class)))[1]:
                self.queries.append(os.path.join(queries_path, query_class, class_dir))
            self.n_task = len(self.queries)
        self.default_sample = None

        self.skeleton = skeleton
        with open(f'action_rec/hpe/assets/skeleton_types.pkl', "rb") as input_file:
            skeleton_types = pickle.load(input_file)
        self.edges = skeleton_types[skeleton]['edges']
        self.indices = skeleton_types[skeleton]['indices'] if given_122 else np.array(range(0, 30))

    def get_sample(self, class_name, ss=False, path=None):
        if not path:
            use_exemplars = (ss and self.exemplars_path)
            sequences = next(os.walk(os.path.join(self.queries_path if not use_exemplars else self.exemplars_path,
                                                  class_name)))[1]
            path = random.sample(sequences, 1)[0]  # Get first random file
            path = os.path.join(self.queries_path, class_name, path)  # Create full path

        imgs = []
        poses = []
        i = 0
        while True:
            # Load pose
            if self.input_type in ["hybrid", "skeleton"]:
                with open(os.path.join(path, f"{i}.pkl"), 'rb') as file:
                    # Load skeleton
                    pose = pickle.load(file)
                pose = pose[self.indices]
                pose -= pose[0]
                poses.append(pose.reshape(-1))
            # Load image
            if self.input_type in ["rgb", "hybrid"]:
                img = cv2.imread(os.path.join(path, f"{i}.png"))
                img = cv2.resize(img, (224, 224))
                img = img / 255.
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                imgs.append(img.swapaxes(-1, -2).swapaxes(-2, -3))
            i += 1
            if i == self.max_l:
                break

        sample = {}
        if self.input_type in ["hybrid", "rgb"]:
            sample["rgb"] = np.stack(imgs) if self.l == self.max_l else np.stack(imgs)[list(range(0, 16, 2))]
        if self.input_type in ["hybrid", "skeleton"]:
            sample["sk"] = np.stack(poses) if self.l == self.max_l else np.stack(poses)[list(range(0, 16, 2))]
            if self.do_augmentation:
                sample['sk'] = rotate_y(sample['sk'].reshape(-1, 30, 3), random.uniform(0, 360)).reshape(-1, 90)
        return sample

    def __getitem__(self, _):  # Must return complete, imp_x and impl_y
        support_classes = random.sample(self.all_classes, self.k) if not self.support_classes else self.support_classes
        target_class = random.sample(support_classes, 1)[0]
        unknown_class = random.sample([x for x in self.all_classes if x not in support_classes], 1)[0]

        support_set = [[self.get_sample(cl, ss=True) for _ in range(self.n)] for cl in support_classes]

        res = []
        for t in support_set[0][0].keys():
            for support in support_set:
                voc = {t: np.stack([elem[t] for elem in support])}
                res.append(voc)
        support_set = res

        target_set = self.get_sample(target_class, path=self.queries[_] if self.queries else None)
        unknown_set = self.get_sample(unknown_class)

        return {'support_set': {t: np.stack([elem[t] for elem in support_set]) for t in support_set[0].keys()},
                'target_set': target_set,
                'unknown_set': unknown_set,
                'support_classes': np.stack([self.all_classes.index(elem) for elem in support_classes]),
                'target_class': self.all_classes.index(target_class),
                'unknown_class': self.all_classes.index(unknown_class),
                'known': target_class in support_classes}

    def __len__(self):
        return self.n_task


if __name__ == "__main__":
    from action_rec.hpe.utils.matplotlib_visualizer import MPLPosePrinter
    from action_rec.ar.utils.configuration import TRXTrainConfig

    loader = MyLoader("/media/sberti/Data/datasets/NTURGBD_to_YOLO_METRO_122", input_type="rgb", given_122=True, l=16, do_augmentation=True)
    vis = MPLPosePrinter()

    for asd in loader:
        sup = asd['support_set']
        trg = asd['target_set']
        unk = asd['unknown_set']
        lab = asd["support_classes"]

        print(asd['support_classes'])
        n_classes, n_examples, n_frames, _ = sup["sk"].shape
        for c in range(n_classes):
            for n in range(n_examples):
                for k in range(n_frames):
                    cv2.imshow("sup", sup["rgb"][c][k].swapaxes(0, 1).swapaxes(1, 2))
                    cv2.waitKey(1)
                    # vis.set_title(f"{loader.all_classes[lab[c]]}, {n}, {k}")
                    # vis.clear()
                    # vis.print_pose(sup["sk"][c][n][k].reshape(-1, 3), loader.edges)
                    # vis.sleep(0.1)
