from human.modules.focus import FocusDetector
from human.modules.ar import ActionRecognizer
from human.modules.hpe import HumanPoseEstimator
from human.utils.params import FocusConfig, MetrabsTRTConfig, RealSenseIntrinsics, TRXConfig
from utils.multiprocessing import Node
import time
import numpy as np
import cv2
from multiprocessing import Queue, Process
from utils.output import VISPYVisualizer


def run_module(module, configurations, input_queue, output_queue):
    x = module(*configurations)
    while True:
        inp = input_queue.get()
        y = x.estimate(inp)
        output_queue.put(y)


class Human(Node):
    def __init__(self):
        super().__init__(name='human')
        self.focus_in = None
        self.focus_out = None
        self.focus_proc = None
        self.hpe_in = None
        self.hpe_out = None
        self.hpe_proc = None
        self.ar = None
        self.window_size = None
        self.fps_s = None
        self.last_poses = None
        self.visualizer = None
        self.input_queue = None
        self.output_proc = None
        self.output_queue = None

    def startup(self):
        # Load modules
        self.focus_in = Queue(1)
        self.focus_out = Queue(1)
        self.focus_proc = Process(target=run_module, args=(FocusDetector,
                                                           (FocusConfig(),),
                                                           self.focus_in, self.focus_out))
        self.focus_proc.start()

        self.hpe_in = Queue(1)
        self.hpe_out = Queue(1)
        self.hpe_proc = Process(target=run_module, args=(HumanPoseEstimator,
                                                         (MetrabsTRTConfig(), RealSenseIntrinsics()),
                                                         self.hpe_in, self.hpe_out))
        self.hpe_proc.start()

        self.ar = ActionRecognizer(TRXConfig())

        self.fps_s = []
        self.last_poses = []

        # Create output
        self.visualizer = True
        if self.visualizer:
            self.input_queue = Queue(1)
            self.output_queue = Queue(1)
            self.output_proc = Process(target=VISPYVisualizer.create_visualizer,
                                       args=(self.output_queue, self.input_queue))
            self.output_proc.start()

    def loop(self, data):
        img = data['rgb']
        start = time.time()

        # Start independent modules
        focus = False

        self.hpe_in.put(img)
        self.focus_in.put(img)

        pose3d_abs, edges, bbox = self.hpe_out.get()
        focus_ret = self.focus_out.get()

        if focus_ret is not None:
            focus, face = focus_ret

        # Compute distance
        d = None
        if pose3d_abs is not None:
            cam_pos = np.array([0, 0, 0])
            man_pose = np.array(pose3d_abs[0])
            d = np.sqrt(np.sum(np.square(cam_pos - man_pose)))

        # Normalize
        pose3d_root = pose3d_abs - pose3d_abs[0, :] if pose3d_abs is not None else None

        # Make inference
        results = self.ar.inference(pose3d_root)

        end = time.time()

        # Compute fps
        self.fps_s.append(1. / (end - start))
        fps_s = self.fps_s[-10:]
        fps = sum(fps_s) / len(fps_s)

        if self.visualizer:
            if pose3d_abs is not None:
                # Send to visualizer
                img = cv2.flip(img, 0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elements = {"img": img,
                            "pose": pose3d_root,
                            "edges": edges,
                            "fps": fps,
                            "focus": focus,
                            "actions": results,
                            "distance": d * 2,  # TODO fix
                            "box": bbox
                            }
                self.output_queue.put((elements,))

        return img, pose3d_root, results

    def shutdown(self):
        pass