
'''
1. Import Libraries for Data Engineering, Model Engineering
'''

from terminaltables import AsciiTable

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.train_utils import *
from torch.autograd import Variable
from eval_mAP import evaluate_mAP

from models.models import *
from config.train_config import parse_train_configs

def main():
    
    '''
    2. Set Hyperparameters w/argparse.
    '''
    # Get data configuration
    configs = parse_train_configs()
    
    print(configs.device)
    
    '''
    3. Data Engineering
    '''
    # Get data configuration
    data_config = parse_data_config(configs.data_config)
    train_path  = data_config["train"]
    valid_path  = data_config["valid"]
    class_names = load_classes(data_config["names"])
    
    train_dataset = ListDataset(train_path, augment=True, multiscale=configs.multiscale_training)
    test_dataset = ListDataset(valid_path, augment=False, multiscale=False)
    # dataset = ListDataset(valid_path, augment=True, multiscale=configs.multiscale_training)
    
    '''
    4. Build NN model
    '''
    
    model = Darknet(configs.model_def)
    # model.apply(weights_init_normal)
    # model.print_network()
    model = model.to(configs.device)
    
    '''
    5. Load Pre-trained data
    '''    
    print(configs.pretrained_path)
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)

    # If specified we start from checkpoint
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            model.load_state_dict(torch.load(configs.pretrained_path))
            print("Trained pytorch weight loaded!")
        else:
            model.load_darknet_weights(configs.pretrained_path)
            print("Darknet weight loaded!")
    # torch.save(model.state_dict(), "checkpoints/Yolo_V3_coco.pth")
    # torch.save(model.state_dict(), "checkpoints/Yolo_V3_coco_tiny.pth")
    
    '''
    6. Optimizer Define
    '''
    optimizer = torch.optim.Adam(model.parameters())
    
    '''
    7. Metrics Define
    '''
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
        # "recall50",
        # "recall75",
        # "precision",
        "conf_obj",
        "conf_noobj",
    ]
    
    '''
    8. learning rate scheduler
    '''
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    '''
    9. DataLoader
    '''
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size, 
        shuffle=True,
        num_workers=configs.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=configs.batch_size, 
        shuffle=False,
        num_workers=configs.n_cpu,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )
    
    '''
    10. Define Loss Fumction
    Yolo Loss is caculated at train step
    '''
    
    '''
    12. Define Episode / each step process
    '''

    max_mAP = 0.0
    start_time = time.time() 
    for epoch in range(0, configs.num_epochs, 1):
        
        '''
        11. Define train loop
        '''
        num_iters_per_epoch = len(train_dataloader)

        print(num_iters_per_epoch)
        
        '''
        Tr 1. Model switch to train mode
        '''
        model.train()
        
        epoch_loss = 0
        
        '''
        # For each step(batch)
        '''
        for batch_idx, batch_data in enumerate(tqdm.tqdm(train_dataloader)):
            """
            Tr 2. Explore Train_dataloader
            Below Code will be used at the detailed Data Engineering
            
            # print(batch_data)
            
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
            
            # End of exploration.
            """
            
            '''
            Tr 3. Transfer Data to GPU if available
            '''
            # data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            global_step = num_iters_per_epoch * epoch + batch_idx + 1
            
            targets = Variable(targets.to(configs.device), requires_grad=False)
            imgs    = Variable(imgs.to(configs.device))
            
            '''
            Tr 4. Forward Pass
            '''
            '''
            Tr 5. Compute Loss
            '''
            total_loss, outputs = model(imgs, targets)
            
            epoch_loss += float(total_loss.item())
            
            '''
            Tr 6. compute gradient and perform backpropagation
            '''            
            total_loss.backward()

            if global_step % configs.gradient_accumulations:
                
                '''
                Tr 7. Accumulates gradient before each step
                '''
                optimizer.step()
                
                '''
                Tr 8. Adjust learning rate if IMPL is using learning rate scheduler
                '''
                lr_scheduler.step()

                '''
                Tr 9. Clear Gradients (zero the parameter gradients)
                '''
                optimizer.zero_grad()
            
            
            """
            Tr 10. [Opt] Metric Table and Log progress 
            at each YOLO layer will be handled with separate chapter.
            """
            if (batch_idx+1) % int(len(train_dataloader)/3) == 0:

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % ((epoch+1), configs.num_epochs, (batch_idx+1), len(train_dataloader))

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
                
            """
            # End of Metric Table and Log progress 
            """
            
        crnt_epoch_loss = epoch_loss/num_iters_per_epoch
        
        """
        Tr 11. [Opt] Save model 
        """
        torch.save(model.state_dict(), configs.save_path)
        # global_epoch += 1
        
        # print("Global_epoch :",global_epoch, "Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        print("Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
    
    '''
    13. Define test loop
    '''
    num_iters_per_epoch = len(test_dataloader)

    print(num_iters_per_epoch)
    
    '''
    Val 1. Model switch to evaluation mode
    '''
    model.eval()

    test_loss = 0.
    
    '''
    # For each step(batch)
    '''
    for batch_idx, batch_data in enumerate(tqdm.tqdm(test_dataloader)):
        '''
        Val 2. Transfer Data to GPU if available
        '''
        # data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * epoch + batch_idx + 1

        targets = Variable(targets.to(configs.device), requires_grad=False)
        imgs    = Variable(imgs.to(configs.device))
        '''
        Val 3. Forward Pass
        '''
        
        '''
        Val 4. Compute Loss
        '''
        total_loss, outputs = model(imgs, targets)

        test_loss += float(total_loss.item())

    test_loss /= len(test_dataloader)

    print("Current epoch loss : {:1.5f}".format(test_loss),'Saved at {}'.format(configs.save_path))
    
    
    '''
    14. Mean Average Precision
    '''
    # Evaulation
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate_mAP(model, valid_path, configs,
        batch_size=configs.batch_size)

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
    
    """
    [Opt]. Save checkpoint
     
    if (epoch+1) % configs.checkpoint_freq == 0:
        torch.save(model.state_dict(), configs.save_path)
        print('save a checkpoint at {}'.format(configs.save_path))
    """

if __name__ == '__main__':
    main()
    