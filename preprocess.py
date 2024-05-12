

#%%
import shutil
import os

# %%
data_dir = "spacecraft_detection/data"

#%%
os.listdir(data_dir)
# %%
dir_0 = "spacecraft_detection/data/0"

#%%

os.listdir(dir_0)

#%%

#folder 8, 9 had lsess images redownload it

# %%
import pandas as pd

#%%
train_label_path = "spacecraft_detection/data/train_labels.csv"

train_metadata_path = "spacecraft_detection/data/train_metadata.csv"
train_df = pd.read_csv(train_label_path)
train_metadata_df = pd.read_csv(train_metadata_path)

#%%
train_df_cp = train_df.copy(deep=True)

#%%
train_df_cp["folder_name"] = train_df_cp["image_id"].str[0]

#%%
train_df_cp.groupby("folder_name")["image_id"].count().reset_index()

#%%

train_metadata_df["spacecraft_id"].value_counts().reset_index()

#%%

train_metadata_df["background_id"].value_counts()
#%%
train_df.image_id.nunique()

#%%

train_df.columns


#%%
all_images_dir = "spacecraft_detection/all_images"

# %%
len(os.listdir(all_images_dir))


# %%
bg_path = "spacecraft_detection/data/no_background/no_background.csv"
# %%
bg_df = pd.read_csv(bg_path)
# %%
bg_df["image_id"].nunique()
# %%
bg_img_dir = "spacecraft_detection/no_background/images"

#%%
len(os.listdir(bg_img_dir))


#%%

all_df = train_df.merge(train_metadata_df, left_on="image_id", right_on="image_id")


#%%

from sklearn.model_selection import train_test_split


#%%

trn, test = train_test_split(all_df, test_size=0.3, random_state=2024,
                             stratify=all_df["spacecraft_id"]
                             )

#%%

trn["spacecraft_id"].value_counts().reset_index()

#%%

test["spacecraft_id"].value_counts().reset_index()


#%%
stratified_trn, stratified_test = train_test_split(all_df, test_size=0.3, random_state=2024,
                                  stratify=["spacecraft_id", "background_id"])



#%%
#import cast
all_df["spacecraft_bg_id"] = all_df["spacecraft_id"].astype(str) + '_' + all_df["background_id"].astype(str)



#%%

all_df.groupby("spacecraft_bg_id")["image_id"].count().reset_index()


#%%

strati_trn_spbgid, strati_test_spbgid = train_test_split(all_df, test_size=0.3,
                                                         random_state=2024,
                                                         stratify=all_df["spacecraft_bg_id"]
                                                         )



#%%
from glob import glob
from PIL import Image
import numpy as np
from torch_snippets import Report
import torch


def preprocess_img(img, device):
    img = torch.Tensor(img).permute(2,0,1)
    return img.to(device).float()
    
class SpaceCraftDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, preprocess_fun=preprocess_img):
        self.df = df
        self.img_dir = img_dir
        self.preprocess_fun = preprocess_fun
        self.img_info = self.df.image_id.unique()
        self.files = glob(f"{self.img_dir}/*")
        
    def __getitem__(self, index):
        image_id = self.img_info[index]
        img_file_path = [img_p for img_p in self.files if
                        os.path.basename(img_p).split(".")[0] == image_id
                        ][0]
        #img_name = os.path.basename(img_file_path).split(".")[0]
        img_df = self.df[self.df["image_id"]==image_id]
        img_bbox = img_df[["xmin", "ymin", "xmax", "ymax"]].values
        img_label = img_df["spacecraft_id"].values.tolist()
        img = Image.open(img_file_path).convert("RGB")
        img = np.array(img)/255
        target = {}
        img_bbox_uint32 = img_bbox.astype(np.uint32).tolist()
        target["boxes"] = torch.Tensor(img_bbox_uint32).float()
        
        target["labels"] = torch.Tensor[img_label].long()
        preprocess_img = self.preprocess_fun(img)
        return preprocess_img, target
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.files)
        
        
#%%
all_img_dir = "spacecraft_detection/all_images/"
trn_dataloader = SpaceCraftDataset(df=trn, img_dir=all_img_dir)        
test_dataloader = SpaceCraftDataset(df=test, img_dir=all_images_dir)        

#%%
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, 
                                                      num_classes=num_classes
                                                      )
    return model

#%% 
def train_batch(inputs, model, optimizer):
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses

@torch.no_grad()
def validate_batch(inputs, model):
    model.train()
    input, targets = inputs
    inputs = list(image.to(device) for image in input)
    targets = [{k: v for k,v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(inputs, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses

#%%
num_classes = all_df["spacecraft_id"].nunique() + 1
model = get_model(num_classes=num_classes).to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005, 
                            momentum=0.9, weight_decay=0.0005
                            )
n_epochs = 5
log = Report(n_epochs)

#%% 
def trigger_train():
    for epoch in n_epochs:
        _n = len(trn_dataloader)
        for ix, inputs in enumerate(trn_dataloader):
            loss, losses = train_batch(inputs, model, optimizer)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = (
                [losses[k] for k in ["loss_classifier", "loss_box_reg",
                                    "loss_objectness", "loss_rpn_box_reg"
                                    ]
                ]
            )
            pos = (epoch + (ix+1)/_n)
            log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss.item(),
                       trn_regr_loss=regr_loss.item(),
                       trn_objectness_loss=loss_objectness.item(),
                       trn_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                       end="\r"
                       )
            
        _n = len(test_dataloader)
        for ix, inputs in enumerate(test_dataloader):
            loss, losses = validate_batch(inputs, model)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = (
                [losses[k] for k in ["loss_classifier", "loss_box_reg", "loss_objectness",
                                     "loss_rpn_box_reg"
                                     ]
                 ]
            )
            pos = (epoch + (ix+1)/_n)
            log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss.item(),
                       val_regr_loss=regr_loss.item(), val_objectness=loss_objectness.item(),
                       val_rpn_box_reg_loss=loss_rpn_box_reg.item(),
                       end="\r"
                       )
        if (epoch+1)%(n_epochs//5)==0:
            log.report_avgs(epoch+1)
    return log

#%%
log = trigger_train()

#%%
log.plot_epochs(["trn_loss", "val_loss"])

#%%
from torchvision.ops import nms
def decode_output(output):
    bbs = output["scores"].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target for target in output["labels"].cpu().detach().numpy()])
    confs = output["scores"].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]
    
    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

#%%
model.eval()
for ix, (images, targets) in enumerate(test_dataloader):
    if ix==3: break
    images = [im for im in images]
    outputs = model(images)
    for ix, output in enumerate(outputs):
        bbs, confs, labels = decode_output(output)
        info = [f"{l}@{c:2f}" for l, c in zip(labels, confs)]
        show(images[ix].cpu().permute(1,2,0), bbs=bbs, texts=labels, sz=5)





#%%
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)

#%%
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
output = model(images, targets)
#%% For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)

#%% optionally, if you want to export the model to ONNX:
torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

# %%
# approaches
# datasplitting
# 1. split data based on spacecfrat_id and background_id

# 2. preprocess data with normalization

# train data on Faster-rcnn with resnet50 on 20 epochs
# save results 
# try image augmentations and see the one that is most effective

#%% try retinanet 


#%%
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold


#%%

import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class SimpleConvNet(nn.Module):
  '''
    Simple Convolutional Neural Network
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(1, 10, kernel_size=3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(26 * 26 * 10, 50),
      nn.ReLU(),
      nn.Linear(50, 20),
      nn.ReLU(),
      nn.Linear(20, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
  
if __name__ == '__main__':
  
  # Configuration options
  k_folds = 5
  num_epochs = 1
  loss_function = nn.CrossEntropyLoss()
  
  # For fold results
  results = {}
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Prepare MNIST dataset by concatenating Train/Test part; we split later.
  dataset_train_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
  dataset_test_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
  dataset = ConcatDataset([dataset_train_part, dataset_test_part])
  
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
    
  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=10, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=10, sampler=test_subsampler)
    
    # Init the neural network
    network = SimpleConvNet()
    network.apply(reset_weights)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      current_loss = 0.0

      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        
        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = network(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)

    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):

        # Get inputs
        inputs, targets = data

        # Generate outputs
        outputs = network(inputs)

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

      # Print accuracy
      print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
      print('--------------------------------')
      results[fold] = 100.0 * (correct / total)
    
  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  print(f'Average: {sum/len(results.items())} %')




# %%
