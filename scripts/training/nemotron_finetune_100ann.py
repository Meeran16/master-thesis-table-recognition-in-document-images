import torch.nn as nn
from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # Experiment name
        self.exp_name = "nemotron_finetune_100ann"

        # Dataset
        self.data_dir = "/home/meeran/thesis_data/dataset_100ann_3class_yolox"
        self.train_ann = "train_nemotron.json"
        self.val_ann = "val_nemotron.json"

        # Important: dataset folder names
        self.train_name = "train"
        self.val_name = "val"

        # Nemotron 3-class setup:
        # category_id 2 = cell
        # category_id 3 = row
        # category_id 4 = column
        self.num_classes = 3

        # Image size
        self.input_size = (1024, 1024)
        self.test_size = (1024, 1024)

        # Training parameters
        self.max_epoch = 50
        self.data_num_workers = 2
        self.batch_size = 2

        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 5
        self.warmup_lr = 0.0001
        self.min_lr_ratio = 0.01
        self.weight_decay = 0.0005
        self.momentum = 0.9

        # Architecture used earlier for Nemotron-compatible YOLOX
        self.depth = 1.0
        self.width = 1.0
        self.act = "silu"

        # Reduced augmentation for document images / small dataset
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.0
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 5.0
        self.translate = 0.1
        self.scale = (0.8, 1.2)
        self.shear = 0.0
        self.perspective = 0.0

        # Evaluation / logging
        self.eval_interval = 10
        self.print_interval = 5
        self.save_history_ckpt = True

        # Confidence/NMS for evaluation
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]

            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
            )

            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
            )

            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model
