from collections import OrderedDict
from .utils.model import TRXOS
import torch
import copy
import pickle as pkl
import os
from .utils.configuration import TRXTrainConfig


class ActionRecognizer:
    def __init__(self, input_type=None, device=None, add_hook=False, final_ckpt_path=None, seq_len=None, way=None,
                 n_joints=None, support_set_path=None, shot=None):
        self.input_type = input_type
        self.device = device

        self.ar = TRXOS(TRXTrainConfig(), add_hook=add_hook)
        # Fix dataparallel
        state_dict = torch.load(final_ckpt_path, map_location=torch.device(0))['model_state_dict']
        state_dict = OrderedDict({param.replace('.module', ''): data for param, data in state_dict.items()})
        self.ar.load_state_dict(state_dict)
        self.ar.cuda()
        self.ar.eval()

        # Now
        self.support_set_data_sk = torch.zeros(way, shot, seq_len, n_joints*3).cuda()
        self.support_set_mask = torch.zeros(way, shot).cuda()
        self.support_set_labels = [None] * way
        self.requires_focus = [None] * way
        self.support_set_features = None

        self.previous_frames = []
        self.seq_len = seq_len
        self.way = way
        self.shot = shot
        self.n_joints = n_joints if input_type == "skeleton" else 0
        self.support_set_path = support_set_path

    def inference(self, data):
        """
        It receives an iterable of data that contains poses, images or both
        """
        if data is None or len(data) == 0:
            return {}, 0, {}

        if len(self.support_set_labels) == 0:  # no class to predict
            return {}, 0, {}

        # Process new frame
        data = {k: torch.FloatTensor(v).cuda() for k, v in data.items()}
        self.previous_frames.append(copy.deepcopy(data))
        if len(self.previous_frames) < self.seq_len:  # few samples
            return {}, 0, {}
        elif len(self.previous_frames) == self.seq_len + 1:
            self.previous_frames = self.previous_frames[1:]  # add as last frame

        # Prepare query with previous frames
        for t in list(data.keys()):
            data[t] = torch.stack([elem[t] for elem in self.previous_frames]).unsqueeze(0)

        # Get SS
        ss = {}
        ss_f = None
        # if self.support_set_features is None:
        ss['sk'] = self.support_set_data_sk.unsqueeze(0)
        # else:
        #     ss_f = self.support_set_features
        labels = self.support_set_mask.unsqueeze(0)
        with torch.no_grad():
            outputs = self.ar(ss, labels, data, ss_features=ss_f)  # RGB, POSES

        # Save support features
        if ss_f is None:
            self.support_set_features = outputs['support_features']

        # Softmax
        true_logits = outputs['logits'][:, torch.any(self.support_set_mask, dim=1)]
        few_shot_result = torch.softmax(true_logits.squeeze(0), dim=0).detach().cpu().numpy()
        open_set_result = outputs['is_true'].squeeze(0).detach().cpu().numpy()

        # Return output
        results = {}
        true_labels = list(filter(lambda x: x is not None, self.support_set_labels))
        for k in range(len(true_labels)):
            results[true_labels[k]] = (few_shot_result[k])
        return results, open_set_result, self.requires_focus

    def remove(self, flag):
        # Compute index to remove
        if flag in self.support_set_labels:
            class_id = self.support_set_labels.index(flag)
            self.support_set_labels[class_id] = None
            self.support_set_mask[class_id] = 0
            self.requires_focus[class_id] = None
            self.support_set_data_sk[class_id] = 0
            self.support_set_features = None
            return True
        else:
            return False

    def train(self, inp, ss_id):
        if inp['flag'] not in self.support_set_labels:
            first_none_pos = self.support_set_labels.index(None)
            self.support_set_labels[first_none_pos] = inp['flag']
        class_id = self.support_set_labels.index(inp['flag'])
        self.support_set_data_sk[class_id][ss_id] = torch.FloatTensor(inp['data']['sk']).cuda()
        self.requires_focus[class_id] = inp['requires_focus']
        self.support_set_mask[class_id][ss_id] = 1
        self.support_set_features = None

    def save(self):
        with open(os.path.join(self.support_set_path, "support_set_labels.pkl"), 'wb') as outfile:
            pkl.dump(self.support_set_labels, outfile)
        with open(os.path.join(self.support_set_path, "support_set_data_sk.pkl"), 'wb') as outfile:
            pkl.dump(self.support_set_data_sk, outfile)
        with open(os.path.join(self.support_set_path, "requires_focus.pkl"), 'wb') as outfile:
            pkl.dump(self.requires_focus, outfile)
        with open(os.path.join(self.support_set_path, "support_set_mask.pkl"), 'wb') as outfile:
            pkl.dump(self.support_set_mask, outfile)
        return "Classes saved successfully in " + self.support_set_path

    def load(self):
        with open(os.path.join(self.support_set_path, "support_set_labels.pkl"), 'rb') as pkl_file:
            self.support_set_labels = pkl.load(pkl_file)
        with open(os.path.join(self.support_set_path, "support_set_data_sk.pkl"), 'rb') as pkl_file:
            self.support_set_data_sk = pkl.load(pkl_file)
        with open(os.path.join(self.support_set_path, "requires_focus.pkl"), 'rb') as pkl_file:
            self.requires_focus = pkl.load(pkl_file)
        with open(os.path.join(self.support_set_path, "support_set_mask.pkl"), 'rb') as pkl_file:
            self.support_set_mask = pkl.load(pkl_file)
        return f"Loaded {len(self.support_set_labels)} classes"
