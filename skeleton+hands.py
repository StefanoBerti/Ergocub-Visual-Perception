from action_rec.ar.utils.dataloader import MyLoader
import cv2


if __name__ == "__main__":
    from action_rec.hpe.utils.matplotlib_visualizer import MPLPosePrinter
    from action_rec.ar.utils.configuration import TRXTrainConfig, ubuntu

    input_type = "skeleton"
    loader = MyLoader("/media/sberti/Data/datasets/NTURGBD_to_YOLO_METRO_122", input_type=input_type, given_122=not ubuntu,
                      do_augmentation=False)

    if input_type in ["skeleton", "hybrid"]:
        vis = MPLPosePrinter()

    for asd in loader:
        sup = asd['support_set']
        trg = asd['target_set']
        unk = asd['unknown_set']
        lab = asd["support_classes"]

        print(asd['support_classes'])
        n_classes, n_examples, n_frames = sup[list(sup.keys())[0]].shape[:3]
        for c in range(n_classes):
            for n in range(n_examples):
                for k in range(n_frames):
                    if input_type in ["rgb", "hybrid"]:
                        cv2.imshow("sup", sup["rgb"][c][n][k].swapaxes(0, 1).swapaxes(1, 2))
                        cv2.waitKey(1)
                    if input_type in ["skeleton", "hybrid"]:
                        vis.set_title(f"{loader.all_classes[lab[c]]}, {n}, {k}")
                        vis.clear()
                        vis.print_pose(sup["sk"][c][n][k], loader.edges)
                        vis.sleep(0.001)
