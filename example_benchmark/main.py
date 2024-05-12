import os
from pathlib import Path
import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from glob import glob
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                                      FasterRCNN, 
                                                      FasterRCNN_ResNet50_FPN_Weights
                                                      )
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from typing import Any, Optional
from torchvision.models._utils import _ovewrite_value_param
from torchvision.ops import misc as misc_nn_ops
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models.resnet import ResNet, BasicBlock,Bottleneck  
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._api import Weights, WeightsEnum, register_model
import time

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, 
                                                      num_classes=num_classes
                                                      )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def decode_output(output, nms_threshold):
    #print(f"decode_output: {output}")
    bbs = output["boxes"].cpu().detach().numpy().astype(np.uint16)
    #print(f"decode bbox: {bbs}")
    labels = np.array([target for target in output["labels"].cpu().detach().numpy()])
    confs = output["scores"].cpu().detach().numpy()
    ixs = nms(boxes=torch.tensor(bbs.astype(np.float32)), 
              scores=torch.tensor(confs), 
              iou_threshold=nms_threshold
              )
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]
    
    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

@torch.no_grad()
def predict_bbox(model, test_dataloader,
                 submission_df,
                device, 
                nms_threshold,
                using_torchscript_weight=True,
                resize_bbox_ratios = (0.21875, 0.175)
                ) -> pd.DataFrame:
    model.eval().to(device)
    print(f"nms_threshold: {nms_threshold}")
    num = 0
    num_no_pred = 0
    images_with_no_pred = []
    for ix, (images, img_paths) in enumerate(test_dataloader):
        num_img = len(images)
        image_id = os.path.basename(img_paths[0]).split(".")[0]
        outputs = model(images)
        #print(f"predict_bbox outputs:{outputs}")
        if using_torchscript_weight:
            outputs = outputs[1]
        for ix, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output, 
                                               nms_threshold=nms_threshold
                                               )
            #print(f"imgage_id: {image_id} bbs: {bbs}, confs: {confs}  seen max below \n")
            try:
                max_conf_index = confs.index(max(confs))
            except ValueError:
                num_no_pred =+ 1
                images_with_no_pred.append(img_paths)
                max_conf_index = 0
                bbs, confs, labels = [[0,0,1,1]], [0], [0]
            bbs, confs, labels = [bbs[max_conf_index]], confs[max_conf_index], labels[max_conf_index]
            print("CHecking resizing")
            if resize_bbox_ratios:
                print(f"doing resize_bbox_ratios: {resize_bbox_ratios}")
                ratio_height, ratio_width = resize_bbox_ratios[0], resize_bbox_ratios[1]
                bbox_lement = bbs[0]
                bbox = [bbox_lement[0] / ratio_width, bbox_lement[1] / ratio_height,
                        bbox_lement[2] / ratio_width, bbox_lement[3] / ratio_height
                        ]
                bbox = [int(bb) for bb in bbox]
            else:
                bbox = [int(x) for x in bbs[0]]
            num += 1
            print(f"count: {num} MAX. bbox: {bbox}, bbs: {bbs} confs: {confs}")
            #print(f"num_img: {num_img}")
            submission_df.loc[image_id] = bbox
    print(f"Total num of no predictions: {num_no_pred}")
    print(f"images_with_no_pred: \n {images_with_no_pred}")
    return submission_df

def predict_and_return_df():
    pass

def preprocess_img(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.float()

class SpacecraftPredictDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir,
                 preprocess_fun=preprocess_img,
                 return_image_path=True,
                 files=None, device="cpu",
                 resize=(224,224)
                 ):
        self.img_dir = img_dir
        if not files:
            self.files = glob(f"{self.img_dir}/*")
        else:
            self.files = files
        self.preprocess_fun = preprocess_fun
        self.return_image_path = return_image_path
        self.device = device
        self.resize = resize
        
    def __getitem__(self, index):
        img_file_path = self.files[index]
        img = cv2.imread(img_file_path)
        if self.resize: 
            resize_height, resize_width = self.resize[0], self.resize[1]
            img = np.resize(img, (resize_height, resize_width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)/255
        preprocess_img = self.preprocess_fun(img)
        if self.return_image_path:
            return preprocess_img.to(self.device), img_file_path
        else:
            return preprocess_img.to(self.device)
        
    def __len__(self):
        return len(self.files)
        
        
    def collate_fn(self, batch):
        return tuple(zip(*batch))


def _resnet_override(block: Type[Union[BasicBlock, Bottleneck]],
                     layers: List[int],
                     weights: Optional[WeightsEnum],
                     weight_path,
                     num_classes=None,
                     device=None,
                     **kwargs: Any,
                    ) -> ResNet:

    model = ResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load(weight_path, 
                                     map_location=torch.device(device))
                          )

    return model


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50_override(*, weights: Optional[ResNet50_Weights] = None,
                      resnetweight_path=None,
                      num_classes=2, device=None,
                      **kwargs
                      ) -> ResNet:
    weights = ResNet50_Weights.verify(weights)

    return _resnet_override(Bottleneck, [3, 4, 6, 3], weights,
                            weight_path=resnetweight_path,
                            num_classes=num_classes, device=device,
                            **kwargs
                            )


def load_model(weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
                progress: bool = True,
                num_classes: Optional[int] = None,
                weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
                trainable_backbone_layers: Optional[int] = None,
                weight_path: str = None,
                resnetweight_path: str = None,
                device = None
                ):
    weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50_override(weights=weights_backbone, 
                                 norm_layer=norm_layer, 
                                 resnetweight_path=resnetweight_path,
                                 num_classes=num_classes,
                                 device=device
                                 )
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    return model

   
@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False),
)
def main(data_dir, output_path):
    # locate key files and locations
    since = time.time()
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    submission_format_path = data_dir / "submission_format.csv"
    images_dir = data_dir / "images"

    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"
    assert output_path.parent.exists(), f"Expected output directory {output_path.parent} does not exist"
    assert submission_format_path.exists(), f"Expected submission format file {submission_format_path} does not exist"
    assert images_dir.exists(), f"Expected images dir {images_dir} does not exist"
    logger.info(f"using data dir: {data_dir}")

    # copy the submission format file; we'll use this as template and overwrite placeholders with our own predictions
    submission_format_df = pd.read_csv(submission_format_path, index_col="image_id")
    submission_df = submission_format_df.copy()
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    #model_path = "spacecraft_onlybbox_epoch_9.pth"
    # model_path = "spacecraft_224X224_epoch_4.pth"
    # resnetweight_path = "resnet50-0676ba61.pth"
    # model = load_model(weight_path=model_path,
    #                     num_classes=2, 
    #                     resnetweight_path=resnetweight_path,
    #                     device=device
    #                     )
    test_data = SpacecraftPredictDataset(img_dir=images_dir)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1,
                                 drop_last=False,
                                 collate_fn=test_data.collate_fn,
                                )
    model = torch.jit.load("spacecraft_fullsize_ssd300_vgg16_torchscript_epoch_1.pt")
    model = torch.jit.script(model)
    submission_df = predict_bbox(model=model, test_dataloader=test_dataloader,
                                 submission_df=submission_df.copy(),
                                 device=device, nms_threshold=0.0001,
                                 using_torchscript_weight=True,
                                 resize_bbox_ratios=None
                                )
    submission_df.to_csv(output_path, index=True)
    time_elapsed = time.time() - since
    print('Prediction Execution completed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    main()

