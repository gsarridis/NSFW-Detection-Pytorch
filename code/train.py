import os

import torch
from torch import nn
import utils.loaders 
from utils.nn_utils import *
import argparse
from datetime import datetime
from torch import optim
from tqdm import tqdm
from torchvision.models.efficientnet import MBConv
from torchvision.ops.misc import ConvNormActivation


def arg_parser():
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--modelname', type=str, default='efficientnet-b1', help='Choose a model', choices=['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4'])
    parser.add_argument('--outpath', type=str, default='./results/models/')
    parser.add_argument('--load_from_cp', type=int, default=0, choices=[0,1], help='Use this option to load a model from a checkpoint')
    parser.add_argument('--modelpath', type=str, default='./results/models/model.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam','sgd'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps_param', type=int, default=0)
    parser.add_argument('--minibatch', type=int, default=-1, help='Use this option to run the exp with a large batch size on limited resources. Activated if >0')

    # Parse the argument
    args = parser.parse_args()
    return args



def freeze_batch_norm(net):
    """Freezes the batch normalization layers

    Args:
        net (_type_): the model

    Returns:
        _type_: the updated model
    """
    print('Freezed batch norm')
    layers = [mod for mod in net.children()]
    for seq_layer in layers[0].children():
        if isinstance(seq_layer, ConvNormActivation):
            
            bn_layer = [mod for mod in seq_layer.children()][1]
            
            # print(f"first layer: {bn_layer}")
            for param in bn_layer.parameters():
                param.requires_grad = False
            
        elif isinstance(seq_layer, nn.Sequential):
            # print('it is')
            for seq_layer2 in seq_layer.children():
                if isinstance(seq_layer2, MBConv):
                    # print('it is 2')
                    block = seq_layer2.block
                    for l in block.children():
                        if isinstance(l, ConvNormActivation):
                            bn_layer = [mod for mod in l.children()][1]
                            # print(bn_layer)
                            for param in bn_layer.parameters():
                                param.requires_grad = False
                    param.requires_grad = False
    return net


def train(net, optimizer, loss_fn, loader, val_loader, epochs, device,  act, exp_name, outpath, base_lr, a, batch_size, start_info):

    vid_acc_old = 0
    step = start_info['step'] + 1
    for epoch in range(start_info['epoch']+1, epochs):
        net.train()
        train_loss = 0
        # correct samples counter
        num_correct = 0
        # true positives counter
        tp = 0
        # total true positives
        ttp = 0
        # true negatives counter
        tn = 0
        # total true negatives
        ttn = 0
        # total samples
        total = 0
        # total steps
        n_total_steps = len(loader)
        
        for i, (inputs, targets,_) in enumerate(tqdm(loader)):
            inputs= inputs.to(device)
            targets = targets.to(device)
            # update lr
            lr = get_lr(step, base_lr, a)
            if lr is None:
                break
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            # if minibatches are activated and its time to backpropagate the gradients
            total += targets.size(0)
            if batch_size==-1 or (total%batch_size==0) or  i==n_total_steps or i==0:
                optimizer.zero_grad()
                
            
            outputs = net(inputs)
            if not act == None:
                outputs = act(outputs)
            targets = targets.to(torch.float32)
            targets = torch.unsqueeze(targets.data, 1)
            mask_nsfw =  targets==1
            mask_sfw =  targets==0

            loss_dist = loss_fn(outputs[mask_nsfw], targets.data[mask_nsfw])
            loss_non_dist = loss_fn(outputs[mask_sfw], targets[mask_sfw])
            alpha = 0.25
            loss = alpha * loss_dist + (1- alpha) * loss_non_dist


            loss.backward()
            
            if batch_size==-1 or (total%batch_size==0) or  i==n_total_steps or i==0:
                optimizer.step()

            train_loss += loss.data.item()

            predicted = torch.round(outputs)
            num_correct += torch.sum(predicted == targets)
            tp += torch.sum(predicted[mask_nsfw] == targets[mask_nsfw])
            tn += torch.sum(predicted[mask_sfw] == targets[mask_sfw])


            ttp += targets[mask_nsfw].shape[0]
            ttn += targets[mask_sfw].shape[0]

            acc = num_correct/total
            acc_tp = tp/ttp
            acc_tn = tn/ttn
            if batch_size==-1 or (total%batch_size==0) or  i==n_total_steps or i==0:
                print(f'training acc {acc.item():.5f}, tp {acc_tp.item():.5f} , tn {acc_tn.item():.5f}, step {step}, lr {lr}')
                step += 1

            
        print(f"epoch: {epoch}, loss = {train_loss/total}, accuracy = {acc}")

        if epoch %1 == 0:
            print("Validation...")
            test_loss, vid_acc = test(net, loss_fn, val_loader, device,act)
            if vid_acc > vid_acc_old:
                save_to_checkpoint(net=net, optimizer=optimizer, epoch=epoch, path=outpath, loss=test_loss, exp_name=exp_name, step=step-1)
                print("Model saved!")
                best_model_dir = os.path.join(outpath,exp_name+'.pt')
                vid_acc_old = vid_acc
        if lr is None:
                break
    net, optimizer, start_info, loss, exp_name = load_from_checkpoint(net,optimizer,best_model_dir)
    net.to(device)
    print(f"Best model loaded. epoch = {epoch}, train loss = {loss}")
    return net

def video_acc(dict, video_list, total_video_frames_dict, default_val, alpha):
    """Calculates the model's accuracy on pornography-2k videos

    Args:
        dict (dict): the dict of videos and their prediction values
        video_list (list): list of the videos
        total_video_frames_dict (dict): the dict of videos and their total number of frames
        default_val (int): if a video in dict has the default value means that it is not included in the test set
        alpha (_type_): the threshold of classifying a video as nsfw or not. if <1, it denotes a proportion, else it denotes a number of frames.

    Returns:
        float: the accuracy 
    """
    val_videos = dict.values()
    val_videos = list(filter(lambda i: i >= 0, val_videos))
    # print(val_videos)
    total_videos = len(val_videos)
    correct_video_predictions = 0
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0
    fp_ids = []
    fn_ids = []
    # th = 10
    for vid in video_list:
        if alpha<1:
            th = total_video_frames_dict[vid] * alpha
        else:
            th = alpha
        if ('vPorn' in vid and dict[vid]>=th):
            correct_video_predictions += 1
            true_pos += 1
        elif ('vPorn' in vid and dict[vid]<th and dict[vid]>default_val):
            false_neg += 1
            fn_ids.append(vid)
        if ('vNonPorn' in vid and dict[vid]<th and dict[vid]>default_val):
            correct_video_predictions += 1
            true_neg += 1
        elif ('vNonPorn' in vid and dict[vid]>=th):
            false_pos += 1
            fp_ids.append(vid)
    print(f"threshold={alpha}, Correct = {correct_video_predictions}, Total videos = {total_videos}, Video Accuracy = {correct_video_predictions/total_videos}, True pos: {true_pos}, True neg: {true_neg}, False pos: {false_pos}, False neg: {false_neg}")
    print(f'Fp ids: {fp_ids}')
    print(f'Fn ids: {fn_ids}')
    return correct_video_predictions/total_videos

def test(net, loss_fn, loader, device, act):
    net.eval()

    test_loss = 0
    num_correct = 0
    total = 0
    tp = 0
    ttp = 0
    tn = 0
    ttn = 0
    #create the video list
    video_list = []
    for cat in ['vNonPorn', 'vPorn']:
        for id in range(1,1001):
            str_id = format(id, '06d')
            video_list.append(cat+str_id)
    # create a dict to keep the predictions
    default_val = -1
    dict = {el:default_val for el in video_list}
    total_video_frames_dict = {el:default_val for el in video_list}
    with torch.no_grad():
        for (inputs, targets,names) in tqdm(loader):
            inputs= inputs.to(device)
            targets = targets.to(device)
            
            outputs = net(inputs)
            if not act == None:
                outputs = act(outputs)

            targets = targets.to(torch.float32)
            targets = torch.unsqueeze(targets.data, 1)
            mask_nsfw =  targets==1
            mask_sfw =  targets==0
            loss = loss_fn(outputs,targets)

            test_loss += loss.data.item()

            predicted = torch.round(outputs)
            num_correct += torch.sum(predicted == targets)
            tp += torch.sum(predicted[mask_nsfw] == targets[mask_nsfw])
            tn += torch.sum(predicted[mask_sfw] == targets[mask_sfw])
            total += targets.size(0)

            # monitor the video predictions
            cc = 0
            for n in names:
                if 'vPorn' in n:
                    if dict[n[:11]] == default_val:
                        dict[n[:11]] = 0
                        total_video_frames_dict[n[:11]] = 0
                    # print(f"Label: {n[:11]}, score:{dict[n[:11]]}")
                    dict[n[:11]] += predicted[cc]
                    total_video_frames_dict[n[:11]] += 1
                else:
                    if dict[n[:14]] == default_val:
                        dict[n[:14]] = 0
                        total_video_frames_dict[n[:14]] = 0
                    # print(f"Label: {n[:14]}, score:{dict[n[:14]]}")
                    dict[n[:14]] += predicted[cc]
                    total_video_frames_dict[n[:14]] += 1 
                cc += 1
            ttp += targets[mask_nsfw].shape[0]
            ttn += targets[mask_sfw].shape[0]

            acc = num_correct/total
            acc_tp = tp/ttp
            acc_tn = tn/ttn
    print(f"tp = {acc_tp}, tn = {acc_tn}, acc = {acc}")
    acc = num_correct/total
    print(f"Loss = {test_loss/total}, Accuracy = {acc}")

    #video accuracy
    _ = video_acc(dict, video_list, total_video_frames_dict, default_val, 1)
    _ = video_acc(dict, video_list, total_video_frames_dict, default_val, 3)
    _ = video_acc(dict, video_list, total_video_frames_dict, default_val, 5)
    _ = video_acc(dict, video_list, total_video_frames_dict, default_val, 0.1)
    vid_acc = video_acc(dict, video_list, total_video_frames_dict, default_val, 0.2)
    _ = video_acc(dict, video_list, total_video_frames_dict, default_val, 0.4)
    _ = video_acc(dict, video_list, total_video_frames_dict, default_val, 0.5)
    _ = video_acc(dict, video_list, total_video_frames_dict, default_val, 0.6)

    return test_loss, vid_acc

def get_lr(step, base_lr=0.003, a=1):
    """Returns learning-rate for `step` or None at the end."""
    # for constant lr
    if a == 0:
        return base_lr
    supports = [500, 3000*a, 6000*a, 9000*a, 10_000*a]
    # Linear warmup
    if step < supports[0]:
        return base_lr * step / supports[0]
    # End of training
    elif step >= supports[-1]:
        return None
    # Staircase decays by factor of 10
    else:
        for s in supports[1:]:
            if s < step:
                base_lr /= 10
        return base_lr

def main():
    _ = ensure_reproducability()
    # Create the parser
    args = arg_parser()
    print(args)

    # Set device
    device = torch.device(args.device)

    # set an experiment name
    exp_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(f"experiment id: {exp_name}")

    # load data
    if args.minibatch == -1:
        full_batch = -1
        train_loader, val_loader = utils.loaders.custom_nsfw_data_loader(batch=args.batch_size)
    else:
        full_batch = args.batch_size
        train_loader, val_loader = utils.loaders.custom_nsfw_data_loader(batch=args.minibatch)
    


    # initialize model and optimizer
    model = load_model(args.modelname, classes=2, isbinary=True, pretrained=True)
    model.to(device)
    # freeze the batch normalization layers
    model = freeze_batch_norm(model)

    # initialize the optimizer
    if args.optimizer == 'adam':
        optimizer =  optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    # load from checkpoint
    if args.load_from_cp:
        model, optimizer, start_info, loss, exp_name = load_from_checkpoint(model,optimizer, args.modelpath)
        model.to(device)
    else:
        start_info = { 'epoch': 0, 'step': 0}
    
    # set loss function and activation function
    loss_fn = nn.BCELoss()
    act= nn.Sigmoid()

    # start training
    if args.epochs>0:
        print("Training...")
        model = train(net=model, optimizer=optimizer, loss_fn=loss_fn, loader=train_loader, val_loader=val_loader, epochs=args.epochs, device=device, act=act, exp_name=exp_name, outpath=args.outpath, base_lr=args.lr, a = args.steps_param, batch_size=full_batch, start_info=start_info)

    print("Testing...")
    test(net=model, loss_fn=loss_fn, loader=val_loader,device=device, act=act)

if __name__ == '__main__':
    main()
    
