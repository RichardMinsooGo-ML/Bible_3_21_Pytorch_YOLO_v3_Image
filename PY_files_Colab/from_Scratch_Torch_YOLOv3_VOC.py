# Check GPU Type
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
    print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
    print('and then re-execute this cell.')
else:
    print(gpu_info)
    

# Memory Space
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
    print('To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
    print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
    print('re-execute this cell.')
else:
    print('You are using a high-RAM runtime!')
    

!pip install terminaltables
!pip install --upgrade --no-cache-dir gdown

from IPython.display import clear_output 
clear_output()

# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Pytorch-Yolov3-Image.git
# ! git pull origin master
! git pull origin main
# Download Darknet Weights
# ! wget https://pjreddie.com/media/files/yolo-voc.weights 
! wget https://pjreddie.com/media/files/yolov3-tiny.weights 
! wget https://pjreddie.com/media/files/yolov3.weights
# ! wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights 
# ! wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

import shutil
shutil.move("/content/yolov3-tiny.weights", "/content/checkpoints")
shutil.move("/content/yolov3.weights", "/content/checkpoints")
# shutil.move("/content/yolov4-tiny.weights", "/content/tmp")
# shutil.move("/content/yolov4.weights", "/content/tmp") 
% rm -rf sample_data


!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# ! mkdir dataset
!tar -xvf VOCtrainval_11-May-2012.tar -C /content/dataset
! rm /content/VOCtrainval_11-May-2012.tar
% rm -rf sample_data

clear_output()


# train valid list file generation
import os

str_train = open("tmp_train.txt", "w")
str_test  = open("tmp_val.txt", "w")

count = 0
for path, subdirs, files in os.walk(r"/content/dataset/VOCdevkit/VOC2012/Annotations"):
    for filename in files:
        f = os.path.join(filename)
        f = os.path.splitext(f)[0]
        
        count += 1
        if count%5 == 0:
            str_test.write("/content/dataset/VOC2012/images/" + str(f) +".jpg"+ os.linesep)
        else:
            str_train.write("/content/dataset/VOC2012/images/" + str(f) +".jpg"+ os.linesep)
        # str_train.write(str(f))
        
with open('tmp_train.txt') as infile, open('/content/dataset/VOC2012/train.txt', 'w') as outfile:
    for line in infile:
        if not line.strip(): continue  # skip the empty line
        outfile.write(line)  # non-empty line. Write it to output
with open('tmp_val.txt') as infile, open('/content/dataset/VOC2012/valid.txt', 'w') as outfile:
    for line in infile:
        if not line.strip(): continue  # skip the empty line
        outfile.write(line)  # non-empty line. Write it to output

! rm tmp_train.txt
! rm tmp_val.txt       

path = "/content/dataset/VOC2012/images/"
os.mkdir(path)
path = "/content/dataset/VOC2012/labels/"
os.mkdir(path)
path = "/content/dataset/VOC2012/xml_files/"
os.mkdir(path)

# move files 
import os
import shutil

source_folder = r"/content/dataset/VOCdevkit/VOC2012/JPEGImages//"
destination_folder = r"/content/dataset/VOC2012/images//"

# fetch all files
for file_name in os.listdir(source_folder):
    # construct full file path
    source = source_folder + file_name
    destination = destination_folder + file_name
    # move only files
    if os.path.isfile(source):
        shutil.move(source, destination)
        print('Moved:', file_name)

source_folder = r"/content/dataset/VOCdevkit/VOC2012/Annotations//"
destination_folder = r"/content/dataset/VOC2012/xml_files//"
# fetch all files
for file_name in os.listdir(source_folder):
    # construct full file path
    source = source_folder + file_name
    destination = destination_folder + file_name
    # move only files
    if os.path.isfile(source):
        shutil.move(source, destination)
        print('Moved:', file_name)

clear_output()

% rm -rf /content/dataset/VOCdevkit


# Create VOC Dataset form
"""
출처: https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py

Pascal VOC의 원본 annotation을 Yolo v3의 Object Detection label format으로 변환해주는 스크립트입니다.
이 파일을 VOCdevkit의 상위폴더로 옮긴 후, 해당 폴더에서 아래 명령을 실행하십시오.
    python voc_label.py

출력되는 파일은 아래와 같습니다.
    2007_test.txt
    train.txt
    voc_classes.txt
2007_test.txt, train.txt 파일을 열어보면 이미지들의 위치가 절대경로로 적혀있습니다.
만약 상대경로로 바꿔주려면 VSCode에서 알맞게 변환해줍니다.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

# classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id):
    in_file  = open('/content/dataset/VOC2012/xml_files/%s.xml'%(image_id))
    out_file = open('/content/dataset/VOC2012/labels/%s.txt'%(image_id), 'w')
    # in_file  = open('COCO2017/Annotations/%s.xml'%(image_id))
    # out_file = open('COCO2017/labels/%s.txt'%(image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        
        bb = np.around(bb, decimals=6)
        
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

sets = [('train'), ('valid')]

if __name__ == '__main__':
    wd = os.getcwd()

    for image_set in sets:
        image_ids = open('/content/dataset/VOC2012/%s.txt' % (image_set)).read().strip().split()
        for image_id in image_ids:
            
            image_id = image_id.split("/")
            image_id = image_id[5]
            image_id = image_id.split(".")
            image_id = image_id[0]
            
            # print(image_id)
            
            convert_annotation(image_id)
        # list_file.close()
          
clear_output()

# Zip Dataset 
# !zip -r /content/file.zip /content/dataset/VOC2012/labels


# Train
from terminaltables import AsciiTable

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets

from torchvision import transforms
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.train_utils import *
from torch.autograd import Variable
import torch.optim as optim
from eval_mAP import evaluate_mAP

from models.models import *


""" configuration json을 읽어들이는 class """
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

config = Config({
    "data_config"  : "/content/config/VOC.data",
    "model_def"    : "/content/config/yolov3.cfg",
    "trained_path" : "/content/checkpoints/yolov3.weights",
    # "saved_path"   : "/content/gdrive/MyDrive/Obj_detection_Torch_img/checkpoints/Yolo_V3_VOC.pth",
    # "trained_path" : "/content/checkpoints/Yolo_V3_VOC.pth",
    "saved_path"   : "/content/checkpoints/Yolo_V3_VOC.pth",
    "working_dir"  : './',
    "num_epochs"   : 3,
    "batch_size"   : 8,
    "grad_accum"   : 2,
    "img_size"     : 416,
    "n_cpu"        : 1
})

print(config)
    
config.eval_interval = 2
config.multiscale_tr = True
config.ckpt_freq     = 2
config.iou_thres     = 0.5
config.conf_thres    = 0.5
config.nms_thres     = 0.5

############## Dataset, logs, Checkpoints dir ######################
config.ckpt_dir = os.path.join(config.working_dir, 'checkpoints')
config.logs_dir = os.path.join(config.working_dir, 'logs')

print(config)

if not os.path.isdir(config.ckpt_dir):
    os.makedirs(config.ckpt_dir)
if not os.path.isdir(config.logs_dir):
    os.makedirs(config.logs_dir)

############## Hardware configurations #############################    
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initiate model
model = Darknet(config.model_def).to(config.device)
# model.apply(weights_init_normal)

# Get data configuration
data_config = parse_data_config(config.data_config)
train_path = data_config["train"]
valid_path = data_config["valid"]

# If specified we start from checkpoint

if config.trained_path:
    if config.trained_path.endswith(".pth"):
        model.load_state_dict(torch.load(config.trained_path))
        print("Trained pytorch weight loaded!")
    else:
        model.load_darknet_weights(config.trained_path)
        print("Darknet weight loaded!")
# torch.save(model.state_dict(), config.trained_path)
# sys.exit()

class_names = load_classes(data_config["names"])

optimizer = torch.optim.Adam(model.parameters())

metrics = [
    "grid_size",
    "loss",
    "loss_x",
    "loss_y",
    "loss_w",
    "loss_h",
    "loss_obj",
    "loss_cls",
    "cls_acc",
    "recall50",
    "recall75",
    "precision",
    "conf_obj",
    "conf_noobj",
]

# learning rate scheduler config
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# Create dataloader
# dataset = ListDataset(train_path, augment=True, multiscale=config.multiscale_tr)
# dataset = ListDataset(valid_path, augment=False, multiscale=False)
dataset = ListDataset(train_path, augment=False, multiscale=False)
# dataset = ListDataset(valid_path, augment=True, multiscale=config.multiscale_tr)

train_dataloader = DataLoader(
    dataset,
    config.batch_size,
    shuffle=True,
    num_workers=config.n_cpu,
    pin_memory=True,
    collate_fn=dataset.collate_fn
)

max_mAP = 0.0
for epoch in range(0, config.num_epochs, 1):

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    start_time = time.time()

    epoch_loss = 0
    # Training        
    for batch_idx, batch_data in enumerate(tqdm.tqdm(train_dataloader)):
        """
        print(batch_data[0])
        print(batch_data[1])
        print(batch_data[1].shape)
        print(batch_data[2])

        imgs = batch_data[1]

        from PIL import Image
        import numpy as np

        w, h = imgs[0].shape[1], imgs[0].shape[2]
        src = imgs[0]
        # data = np.zeros((h, w, 3), dtype=np.uint8)
        # data[256, 256] = [255, 0, 0]

        data = np.zeros((h, w, 3), dtype=np.uint8)
        data[:,:,0] = src[0,:,:]*255
        data[:,:,1] = src[1,:,:]*255
        data[:,:,2] = src[2,:,:]*255
        # img = Image.fromarray(data, 'RGB')
        img = Image.fromarray(data)
        img.save('my_img.png')
        img.show()
        """

        # data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * epoch + batch_idx + 1

        targets = Variable(targets.to(config.device), requires_grad=False)
        imgs = Variable(imgs.to(config.device))

        total_loss, outputs = model(imgs, targets)

        epoch_loss += float(total_loss.item())
        # compute gradient and perform backpropagation
        total_loss.backward()

        if global_step % config.grad_accum:
            # Accumulates gradient before each step
            optimizer.step()
            # Adjust learning rate
            lr_scheduler.step()

            # zero the parameter gradients
            optimizer.zero_grad()

        # ----------------
        #   Log progress
        # ----------------
        """
        if (batch_idx+1)%int((len(train_dataloader)/4)) == 0:

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % ((epoch+1), config.num_epochs, (batch_idx+1), len(train_dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", total_loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, global_step)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {total_loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_dataloader) - (batch_idx + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_idx + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

        # model.seen += imgs.size(0)
        """
    
    crnt_epoch_loss = epoch_loss/num_iters_per_epoch

    if (epoch+1)%3 == 0:
        torch.save(model.state_dict(), config.saved_path)
        print('Saved at {}'.format(config.saved_path))
    # global_epoch += 1

    # print("Global_epoch :",global_epoch, "Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(config.trained_path))
    print("Current epoch loss : {:1.5f}".format(crnt_epoch_loss))


# Evaulation        
#-------------------------------------------------------------------------------------

# if (epoch+1)%8 == 0:
print("\n---- Evaluating Model ----")
# Evaluate the model on the validation set
precision, recall, AP, f1, ap_class = evaluate_mAP(model, valid_path, config,
    batch_size=4)

val_metrics_dict = {
    'precision': precision.mean(),
    'recall': recall.mean(),
    'AP': AP.mean(),
    'f1': f1.mean(),
    'ap_class': ap_class.mean()
}

# Print class APs and mAP
ap_table = [["Index", "Class name", "AP"]]
for i, c in enumerate(ap_class):
    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
print(AsciiTable(ap_table).table)
print(f"---- mAP {AP.mean()}")

max_mAP = AP.mean()
#-------------------------------------------------------------------------------------
"""
# Save checkpoint
if (epoch+1) % config.ckpt_freq == 0:
    torch.save(model.state_dict(), config.trained_path)
    print('save a checkpoint at {}'.format(config.trained_path))
"""

# Test images
from __future__ import division

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from pathlib import Path
Path("output/images").mkdir(parents=True, exist_ok=True)
import cv2

config.image_folder = "dataset/custom/images"
config.class_path  = "/content/dataset/VOC2012/voc2012.names"
config.batch_size = 1
config.conf_thres = 0.8
config.nms_thres  = 0.4

# Set up model
classes = load_classes(config.class_path)
# model.print_network()
print("\n\n" + "-*=" * 30 + "\n\n")
assert os.path.isfile(config.trained_path), "No file at {}".format(config.trained_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config.trained_path:
    if config.trained_path.endswith(".pth"):
        model.load_state_dict(torch.load(config.trained_path))
        print("Trained pytorch weight loaded!")
    else:
        model.load_darknet_weights(config.trained_path)
        print("Darknet weight loaded!")
        
os.makedirs("output", exist_ok=True)
# Eval mode
model.eval()

dataloader = DataLoader(
    ImageFolder(config.image_folder, img_size=config.img_size),
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.n_cpu,
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print("\nPerforming object detection:")
start_time = time.time()
for batch_idx, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections 
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, config.conf_thres, config.nms_thres)

    # Log progress
    end_time = time.time()
    inference_time = datetime.timedelta(seconds=end_time - start_time)
    start_time = end_time
    print("\t+ Batch %d, Inference Time: %s" % (batch_idx, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print("\nSaving images:")

# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, config.img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="yellow", facecolor="none")

            # Add the bbox to the plot
            ax.add_patch(bbox)

            # Add label
            plt.text(x1,y1,s=classes[int(cls_pred)],color="white",verticalalignment="top",bbox={"color": 'C0', "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = path.split("/")[-1].split(".")[0]

    plt.savefig(f"output/pred_{filename}.jpg", bbox_inches="tight", pad_inches=0.0)
    plt.close()

# mAP Calculation
import numpy as np

config.batch_size  = 8
config.n_cpu  = 4
config.iou_thres  = 0.5
config.conf_thres = 0.5
config.nms_thres  = 0.5


# Get data configuration
data_config = parse_data_config(config.data_config)
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

# model.print_network()
print("\n" + "___m__@@__m___" * 10 + "\n")

print(config.trained_path)

assert os.path.isfile(config.trained_path), "No file at {}".format(config.trained_path)

# If specified we start from checkpoint
if config.trained_path:
    if config.trained_path.endswith(".pth"):
        model.load_state_dict(torch.load(config.trained_path))
        print("Trained pytorch weight loaded!")
    else:
        model.load_darknet_weights(config.trained_path)
        print("Darknet weight loaded!")

print(valid_path)
print("\nStart computing mAP...\n")
precision, recall, AP, f1, ap_class = evaluate_mAP(model, valid_path, config, batch_size = config.batch_size)

print("\nDone computing mAP...\n")
for idx, cls in enumerate(ap_class):
    print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
            class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

print("\nmAP: {:.4}\n".format(AP.mean()))


