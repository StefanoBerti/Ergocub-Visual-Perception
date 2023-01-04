import copy
import sys
from multiprocessing.managers import BaseManager
from pathlib import Path

import tensorrt  # TODO NEEDED IN ERGOCUB, NOT NEEDED IN ISBFSAR

sys.path.insert(0,  Path(__file__).parent.parent.as_posix())
from configs.action_rec_config import Network, HPE, FOCUS, AR, MAIN, Logging  # TODO FIX
# from pathlib import Path

import os
import numpy as np
import time
import cv2
from multiprocessing import Process, Queue



docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)


class ISBFSAR(Network.node):
    def __init__(self, input_type, cam_width, cam_height, window_size, skeleton_scale, acquisition_time):
        super().__init__(**Network.Args.to_dict())
        self.input_type = input_type
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.window_size = window_size
        self.fps_s = []
        self.last_poses = []
        self.skeleton_scale = skeleton_scale
        self.acquisition_time = acquisition_time
        self.edges = None
        self.focus_in = None
        self.focus_out = None
        self.focus_proc = None
        self.hpe_in = None
        self.hpe_out = None
        self.hpe_proc = None
        self.ar = None
        self.last_data = None
        self.commands_queue = None
        self.last_log = None

    def startup(self):
        # Load modules
        self.focus_in = Queue(1)
        self.focus_out = Queue(1)
        self.focus_proc = Process(target=run_module, args=(FOCUS, self.focus_in, self.focus_out))
        self.focus_proc.start()

        self.hpe_in = Queue(1)
        self.hpe_out = Queue(1)
        self.hpe_proc = Process(target=run_module, args=(HPE, self.hpe_in, self.hpe_out))
        self.hpe_proc.start()

        self.ar = AR.model(**AR.Args.to_dict())
        self.ar.load()

        # To receive human commands
        BaseManager.register('get_queue')
        manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
        manager.connect()
        self.commands_queue = manager.get_queue("human_console_commands")

    def get_frame(self, img=None, log=None):
        """
        get frame, do inference, return all possible info
        keys: img, bbox, img_preprocessed, human_distance, pose, edges, actions, is_true, requires_focus, focus, face_bbox,
        fps
        """
        start = time.time()
        elements = copy.deepcopy(Logging.keys)

        ar_input = {}

        # If img is not given (not a video), try to get img
        if img is None:
            img = self._recv()["rgb"]
        elements["rgb"] = img

        # Start independent modules
        self.focus_in.put(img)
        self.hpe_in.put(img)

        # RGB CASE
        hpe_res = self.hpe_out.get()
        if self.input_type == "hybrid" or self.input_type == "rgb":
            if hpe_res is not None:
                x1, x2, y1, y2 = hpe_res['bbox']
                elements["bbox"] = x1, x2, y1, y2
                xm = int((x1 + x2) / 2)
                ym = int((y1 + y2) / 2)
                l = max(xm - x1, ym - y1)
                img_ = img[(ym - l if ym - l > 0 else 0):(ym + l), (xm - l if xm - l > 0 else 0):(xm + l)]
                img_ = cv2.resize(img_, (224, 224))
                # cv2.imshow("", img_)  # TODO REMOVE DEBUG
                # cv2.waitKey(1)  # TODO REMOVE DEBUG
                img_ = img_ / 255.
                img_ = img_ * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_ = img_.swapaxes(-1, -3).swapaxes(-1, -2)
                ar_input["rgb"] = img_
                elements["img_preprocessed"] = img_

        # SKELETON CASE
        if self.input_type == "hybrid" or self.input_type == "skeleton":
            if hpe_res is not None:
                pose, edges, bbox = hpe_res['pose'], hpe_res['edges'], hpe_res['bbox']
                if self.edges is None:
                    self.edges = edges
                if pose is not None:
                    elements["human_distance"] = np.sqrt(np.sum(np.square(np.array([0, 0, 0]) - np.array(pose[0])))) * 2.5
                    pose = pose - pose[0, :]
                    elements["pose"] = pose
                    ar_input["sk"] = pose.reshape(-1)
                elements["edges"] = edges
                if bbox is not None:
                    elements["bbox"] = bbox

        # Make inference
        results = self.ar.inference(ar_input)
        actions, is_true, requires_focus = results
        elements["actions"] = actions
        elements["is_true"] = is_true
        elements["action"] = list(actions.keys()).index(max(actions, key=actions.get)) if is_true > 0.66 else -1  # TODO PARAMETRIZE
        elements["requires_focus"] = requires_focus

        # FOCUS #######################################################
        focus_ret = self.focus_out.get()
        if focus_ret is not None:
            focus, face = focus_ret
            elements["focus"] = focus
            elements["face_bbox"] = face.bbox.reshape(-1)

        end = time.time()

        # Compute fps
        self.fps_s.append(1. / (end - start))
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)
        elements["fps"] = fps

        # Msg
        if log is not None:
            self.last_log = log
        elements["log"] = self.last_log

        return elements

    def loop(self, data):
        log = None

        if "rgb" in data.keys():  # Save last data with image
            self.last_data = data
        else:  # It arrives just a message, but we need all
            data.update(self.last_data)

        if not self.commands_queue.empty():
            msg = self.commands_queue.get()
            msg = msg.strip()
            msg = msg.split()

            # select appropriate command
            if msg[0] == 'close' or msg[0] == 'exit' or msg[0] == 'quit' or msg[0] == 'q':
                exit()

            elif msg[0] == "add" and len(msg) > 1:
                log = self.learn_command(msg[1:])
                data = self._recv()

            elif msg[0] == "remove" and len(msg) > 1:
                log = self.forget_command(msg[1])

            elif msg[0] == "save":
                log = self.ar.save()

            elif msg[0] == "load":
                log = self.ar.load()

            elif msg[0] == "debug":
                log = self.debug()
            else:
                log = "Not a valid command!"
        d = self.get_frame(img=data["rgb"], log=log)
        return d

    def forget_command(self, flag):
        if self.ar.remove(flag):
            return "Action {} removed".format(flag)
        else:
            return "Action {} is not in the support set".format(flag)

    def debug(self):
        ss = self.ar.support_set
        if len(ss) == 0:
            return
        if self.input_type in ["hybrid", "imgs"]:
            ss = np.stack([ss[c]["imgs"].detach().cpu().numpy() for c in ss.keys()])
            ss = ss.swapaxes(-2, -3).swapaxes(-1, -2)
            ss = (ss - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            ss = (ss * 255).astype(np.uint8)
            n = len(ss)
            cv2.imshow("support_set_RGB",
                       cv2.resize(ss.swapaxes(0, 1).reshape(8, 224 * n, 224, 3).swapaxes(0, 1).reshape(n * 224, 8 * 224, 3),
                                  (640, 96 * len(ss))))
        if self.input_type in ["hybrid", "skeleton"]:
            ss = np.stack([ss[c]["poses"].detach().cpu().numpy() for c in ss.keys()])
            ss = ss.reshape(ss.shape[:-1]+(30, 3))
            size = 100
            visual = np.zeros((size*ss.shape[0], size*ss.shape[1]))
            ss = (ss + 1)*(size/2)  # Send each pose from [-1, +1] to [0, size]
            ss = ss[..., :2]
            ss[..., 1] += np.arange(ss.shape[0])[..., None, None].repeat(ss.shape[1], axis=1)*size
            ss[..., 0] += np.arange(ss.shape[1])[None, ..., None].repeat(ss.shape[0], axis=0)*size
            ss = ss.reshape(-1, 30, 2).astype(int)
            for pose in ss:
                for point in pose:
                    visual = cv2.circle(visual, point, 1, (255, 0, 0))
                for edge in self.edges:
                    visual = cv2.line(visual, pose[edge[0]], pose[edge[1]], (255, 0, 0))
                cv2.imwrite("SUPPORT_SET.png", visual)
            # cv2.imshow("support_set_SK", visual)
        # cv2.waitKey(0)
        return "Support saved to SUPPORT_SET.png"

    def learn_command(self, flag):
        requires_focus = "-focus" in flag
        flag = flag[0]
        now = time.time()
        while (time.time() - now) < 3:
            self._send_all(self.get_frame(log="WAIT..."), False)

        self._send_all(self.get_frame(log="GO!"), False)
        # playsound('assets' + os.sep + 'start.wav')
        data = [[] for _ in range(self.window_size)]
        i = 0
        off_time = (self.acquisition_time / self.window_size)
        while i < self.window_size:
            start = time.time()
            res = self.get_frame(log="{:.2f}%".format((i / (self.window_size - 1)) * 100))
            self._send_all(res, False)
            # Check if the sample is good w.r.t. input type
            good = self.input_type in ["skeleton", "hybrid"] and "pose" in res.keys() and res["pose"] is not None
            good = good or self.input_type == "rgb"
            if good:
                if self.input_type in ["skeleton", "hybrid"]:
                    data[i].append(res["pose"].reshape(-1))  # CAREFUL with the reshape
                if self.input_type in ["rgb", "hybrid"]:
                    data[i].append(res["img_preprocessed"])
                i += 1
            while (time.time() - start) < off_time:  # Busy wait
                continue

        inp = {"flag": flag,
               "data": {},
               "requires_focus": requires_focus}

        if self.input_type == "rgb":  # Unique case with images in first position
            inp["data"]["imgs"] = np.stack([x[0] for x in data])
        if self.input_type in ["skeleton", "hybrid"]:
            inp["data"]["poses"] = np.stack([x[0] for x in data])
        if self.input_type == "hybrid":
            inp["data"]["imgs"] = np.stack([x[1] for x in data])
        self.ar.train(inp)
        return "Action " + flag + " learned successfully!"


def run_module(module, input_queue, output_queue):
    import pycuda.autoinit
    x = module.model(**module.Args.to_dict())
    while True:
        inp = input_queue.get()
        y = x.estimate(inp)
        output_queue.put(y)


if __name__ == "__main__":
    m = ISBFSAR(**MAIN.Args.to_dict())
    m.run()
