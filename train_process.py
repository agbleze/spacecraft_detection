#%%
from dataset_obj import SpaceCraftDataset
from train import (get_model, trigger_train, predict, 
                   fasterrcnn_resnet50_fpn_load_from_local,
                   train_ssd
                   )
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_snippets import Report
from torchsummary import summary
import cv2
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader
from train import predict, predict_img
from glob import glob
import numpy as np
import torchvision

#%%
train_label_path = "spacecraft_detection/data/train_labels.csv"

train_metadata_path = "spacecraft_detection/data/train_metadata.csv"
train_df = pd.read_csv(train_label_path)
train_metadata_df = pd.read_csv(train_metadata_path)

#%%
all_df = train_df.merge(train_metadata_df,
                        left_on="image_id",
                        right_on="image_id"
                        )

all_df["spacecraft_bg_id"] = all_df["spacecraft_id"].astype(str) + '_' + all_df["background_id"].astype(str)

all_df["only_spacecraft"] = "spacecraft"
trn, test = train_test_split(all_df, test_size=0.3, random_state=2024,
                             stratify=all_df["spacecraft_id"]
                             )


val, test  = train_test_split(test, test_size=0.1, random_state=2024,
                             stratify=test["spacecraft_id"]
                             )
# %%
all_img_dir = "spacecraft_detection/all_images/"
trn_data = SpaceCraftDataset(df=trn, img_dir=all_img_dir, only_bbox=True,
                             resize=None, return_image_path=False
                             )        
val_data = SpaceCraftDataset(df=val, img_dir=all_img_dir, only_bbox=True,
                             resize=None, return_image_path=False
                             )
batch_size = 10
trn_dataloader = DataLoader(dataset=trn_data, batch_size=batch_size, 
                            drop_last=True,
                            collate_fn=trn_data.collate_fn,
                            num_workers=4, pin_memory=True
                            )
val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, 
                            drop_last=True,
                            collate_fn=val_data.collate_fn,
                            num_workers=4, pin_memory=True
                            )

# %%
model_name = "ssd300_vgg16"
num_classes = all_df["spacecraft_id"].nunique() + 1
ssdkwargs = {"detections_per_img":1}
model = get_model(num_classes=2, backbone=model_name,
                  trainable_backbone_layers=5,
                  **ssdkwargs
                  )
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0005,
                            momentum=0.9, weight_decay=0.0005
                            )
n_epochs = 10
log = Report(n_epochs)

#%%

#ssdmodel = torchvision.models.detection.ssd300_vgg16(pretrained=True)

#ssdbackbone = ssdmodel.backbone
#%%
#from torchvision.models.detection._utils import retrieve_out_channels
#from torchvision.models.detection.ssd import SSDHead


#%%
#num_anchors = ssdmodel.anchor_generator.num_anchors_per_location()
#%%
# out_channels = retrieve_out_channels(model=ssdbackbone, size=(1300,1300))
# ssdkwargs = {"detections_per_img":1}
# head = SSDHead(in_channels=out_channels, num_anchors=num_anchors,
#                num_classes=num_classes
#                )

# %%
model_dir = "model_store"

#%%
log = train_ssd(trn_dataloader=trn_dataloader,
                test_dataloader=val_dataloader,
                model=model,
                optimizer=optimizer,
                n_epochs=n_epochs,
                log=log, model_store_dir=model_dir,
                model_name=f"spacecraft_fullsize_{model_name}"
                )

#%%
log = trigger_train(trn_dataloader=trn_dataloader, test_dataloader=val_dataloader,
                    model=model, optimizer=optimizer, n_epochs=n_epochs,
                    log=log, model_store_dir=model_dir,
                    model_name=f"spacecraft_fullsize_{model_name}",
                    )

#%%

imgrd = cv2.imread("spacecraft_detection/all_images/0001954c9f4a58f7ac05358b3cda8d20.png")


#%%
np.resize(imgrd, (224,224))#.shape

#np.array()
#imgrd.resize(224, 224)
#%%
log.plot_epochs(["trn_loss", "val_loss"])

#%%
img_df = all_df[all_df["image_id"]=="0001954c9f4a58f7ac05358b3cda8d20"]
img_bbox = img_df[["xmin", "ymin", "xmax", "ymax"]].values[0]
resize_height, resize_width = 224, 224
ratio_height = resize_height / 1024
ratio_width = resize_width / 1280
img_bbox
#%%
img_bbox = np.float64(img_bbox)
img_bbox
#%%
img_bbox[[0,2]] = np.float64(img_bbox[[0,2]]) * ratio_width #img_bbox[0]* ratio_width, img_bbox[2]* ratio_width
img_bbox[[1,3]] = np.float64(img_bbox[[1,3]]) * ratio_height #img_bbox[1] * ratio_height, img_bbox[3] * ratio_height
img_bbox
img_bbox

#%%


#%%
model_path = "model_store/spacecraft_onlybbox_epoch_9.pth"
# model_loaded = torch.load(model_path)
#%%
model.load_state_dict(torch.load(model_path))
#%%
test_prediction_dir = "test_predictions"
test_data = SpaceCraftDataset(df=test, img_dir=all_img_dir,
                              only_bbox=True,
                              return_image_path=True
                              )

test_dataloader = DataLoader(dataset=test_data, batch_size=1, 
                             drop_last=False,
                            collate_fn=test_data.collate_fn,
                            
                            )

#%%
wrong_pred_dir = "wrong_detections"
nms_pred0001_dir = "nms_pred0001"
nms_pred0001_max_conf = "nms_pred0001_max_conf"
max_conf_pred = "max_conf_pred"

#import torch
#model = torch.load("model_store/spacecraft_epoch_9.pth")

#%%

#from train import fasterrcnn_resnet50_fpn_load_from_local

from train import fasterrcnn_resnet50_fpn_load_from_local

#%%
predict(model=model, test_dataloader=test_dataloader, 
        num_images=None,
        device="cuda",
        save_dir=nms_pred0001_max_conf,
        multiple_pred_save_dir=max_conf_pred,
        nms_threshold=0.0001
        )

#%%
img_paths = glob(f"{all_img_dir}/*")
test_ids = test["image_id"].values.tolist()
#%%
test_img_file_path = [img_p for img_p in img_paths if
                        os.path.basename(img_p).split(".")[0] in test_ids
                        ]#[0]

#%%
predict_img(img_path_list=test_img_file_path,
            model=model, device="cpu",
            export_dir=test_prediction_dir
            )
#%%
from train import predict_img
#%%

import torch

model = torch.load("model_store/spacecraft_epoch_9.pth")

#%%

backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)
model.load_state_dict(torch.load("model_store/spacecraft_epoch_9.pth")
)

#%%

model_name='spacecraft_epoch_9.pth'

model = torch.hub.load("model_store", 
                       'custom', source='local', path = model_name, force_reload = True)


#%%

# approach - loop through the children and 

# %%
img_path = "data/images/0a6ee0963c07afa1d14d77567b35dbb1.png"

Image.open(img_path).convert("RGB").size
# %%
img = cv2.imread(img_path)
# %%
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#.shape
# %%
img_array = np.array(img)#.shape
# %%
torch.tensor(img_array).permute(2,0,1).dim()
# %%
"""
# TO DO.
1. make predictions in a way that test model same as competititom


({}, [{'boxes': tensor([[345.9404, 439.1671, 383.5490, 462.1679],
        [348.7204, 440.4106, 369.5950, 460.7622]]), 'labels': tensor([1, 1]), 'scores': tensor([0.6415, 0.2845])}])
code/__torch__/torchvision/models/detection/faster_rcnn.py:103: UserWarning: RCNN always returns a (Losses, Detections) tuple in scripting

"""