import argparse
import os
import pdb
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
# from torchsummary import summary
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from tensorboardX import SummaryWriter

my_whole_seed = 111
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False

class_weights = None

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('dataset', metavar='Dataset')
parser.add_argument('model_dir', metavar='savedir')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--decay_epoch', default=15, type=int)
parser.add_argument('--seed', default=111, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument("--invalid", action="store_true")

parser.add_argument("--multitask", action="store_true")
parser.add_argument("--liu", action="store_true")
parser.add_argument("--chen", action="store_true")
parser.add_argument("--crossCBAM", action="store_true")
parser.add_argument("--CAN_TS", action="store_true")
parser.add_argument("--crosspatialCBAM", action="store_true")
parser.add_argument("--adam", action="store_true")
parser.add_argument("--choice", default="", type=str)

parser.add_argument("--net_type", default="regular", type=str)
parser.add_argument("--channels", default=109, type=int)
parser.add_argument("--nodes", default=32, type=int)
parser.add_argument("--graph_model", default="WS", type=str)
parser.add_argument("--K", default=4, type=int)
parser.add_argument("--P", default=0.75, type=float)

parser.add_argument("--fold_name", default="", type=str)

# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--base_lr", default=0.03, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)
parser.add_argument("--lambda_value", default=0.25, type=float)
# parser.add_argument('--momentum', default=5e-5, type=float, metavar='M',
#                     help='momentum')

parser.add_argument("--custom_eval", action="store_true")
parser.add_argument("--custom_weights", action="store_true")
parser.add_argument("--custom_semi_predict", action="store_true")

best_acc1 = 0
best_auc = 0
best_accdr = 0
minimum_loss = 1.0
count = 0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parser.parse_args()

    main_worker(args.gpu, args)


def worker_init_fn(worker_id):
    random.seed(1 + worker_id)


def main_worker(gpu, args):
    global best_acc1
    global best_auc
    global minimum_loss
    global count
    global best_accdr
    global class_weights

    args.gpu = gpu

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.arch == "resnet50":
        from models.resnet50 import resnet50
        model = resnet50(num_classes=args.num_class, multitask=args.multitask, liu=args.liu,
                         chen=args.chen, CAN_TS=args.CAN_TS, crossCBAM=args.crossCBAM,
                         crosspatialCBAM=args.crosspatialCBAM, choice=args.choice)

    if args.custom_weights:
        print("Using custom weights")
        class_weights = torch.FloatTensor([1.1465149359886202, 1.9962848297213625, 11.003412969283277, 11.681159420289855, 13.0, 18.528735632183906,
             13.433333333333334, 4.0810126582278485])

    if args.pretrained:
        print("==> Load pretrained model")
        model_dict = model.state_dict()
        pretrain_path = {"resnet50": "pretrain/resnet50-19c8e357.pth",
                         "resnet34": "pretrain/resnet34-333f7ec4.pth",
                         "resnet18": "pretrain/resnet18-5c106cde.pth",
                         "densenet161": "pretrain/densenet161-8d451a50.pth",
                         "vgg11": "pretrain/vgg11-bbd30ac9.pth",
                         "densenet121": "pretrain/densenet121-a639ec97.pth"}[args.arch]
        pretrained_dict = torch.load(pretrain_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict.pop('classifier.weight', None)
        pretrained_dict.pop('classifier.bias', None)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    # summary(model, (3, 224, 224))
    # model = model()
    # model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion)
    if args.dataset == "ODIR" or args.dataset == "ODIR_EXT":
        criterion = nn.MultiLabelSoftMarginLoss(weight=class_weights).cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # define optimizer
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), args.base_lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location={'cuda:4': 'cuda:0'})
            # args.start_epoch = checkpoint['epoch']

            #  load partial weights
            if not args.evaluate:
                print("load partial weights")
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                print("load whole weights")
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    size = 224
    tra = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90),
        # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.ToTensor(),
        normalize,
    ])
    tra_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    # tra = transforms.Compose([
    #     transforms.Resize(350),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    #     transforms.RandomRotation([-180, 180]),
    #     transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
    #     transforms.RandomCrop(224),
    #     #            transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize
    # ])

    # IDRiD dataset
    # tra = transforms.Compose([
    #     transforms.Resize(350),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    #     transforms.RandomRotation([-180, 180]),
    #     transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
    #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # tra_test = transforms.Compose([
    #     transforms.Resize(350),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize])

    if args.dataset == 'amd':
        from datasets.amd_dataset import traindataset
    elif args.dataset == 'pm':
        from datasets.pm_dataset import traindataset
    elif args.dataset == "drdme":
        from datasets.drdme_dataset import traindataset
    elif args.dataset == "missidor":
        from datasets.missidor import traindataset
    elif args.dataset == "ODIR":
        from datasets.ODIR import traindataset
    elif args.dataset == "ODIR_EXT":
        from datasets.ODIR_EXT import traindataset
    elif args.dataset == "kaggle":
        from datasets.kaggle import traindataset
    else:
        print("no dataset")
        exit(0)

    val_dataset = traindataset(root=args.data, mode='val',
                               transform=tra_test, num_class=args.num_class,
                               multitask=args.multitask, args=args)

    train_dataset = traindataset(root=args.data, mode='train', transform=tra, num_class=args.num_class,
                                 multitask=args.multitask, args=args)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.custom_semi_predict:
        #  custom mode for predict the pseudo labels for the messidor dataset!
        from PIL import Image
        import xlrd
        model.eval()

        rootdir = '/scratch_net/petzi/garciaal/git/CANet/data'

        model.cuda()

        pseudo_labels = {}
        list_subfolders_with_paths = [f.path for f in os.scandir(rootdir) if f.is_dir()]

        for dir in list_subfolders_with_paths:
            if "Base" not in dir:
                continue
            base_name = dir.split("/")[-1]
            print("Doing subdir ", base_name)
            directory = "/scratch_net/petzi/garciaal/git/CANet/data/" + base_name

            with torch.no_grad():
                for filename in os.listdir(directory):
                    if filename.endswith(".tif") or filename.endswith(".png"):
                        path = os.path.join(directory, filename)
                        img = Image.open(path)
                        img = img.convert('RGB')

                        img = tra(img)
                        output = model(img.unsqueeze(0).cuda())
                        result = torch.sigmoid(output)
                        # thresholding with 0.25 and 0.75, meaning only taking samples with high confidence
                        #result_F_HC = result < 0.25
                        #result_T_HC = result > 0.75
                        #result_test = result_F_HC + result_T_HC
                        #if result_test.sum() != 8:
                            # Set all result to False, so that it won't take them in ODIR.sh
                        #    result = result > 2
                        #else:
                            # This value only is important when the sum is not 8
                        #    result = result > 0.5
                        result = result > 0.5
                        pseudo_labels[path] = result.cpu().numpy()[0]
                    else:
                        continue
        np.save("/scratch_net/petzi/garciaal/git/CANet/" + os.path.dirname(format(args.resume))  + "/pseudo_labels.npy", pseudo_labels)
        return

    if args.custom_eval:
        #  custom mode for evaluating the pseudo labels for the messidor dataset!
        from PIL import Image
        import xlrd
        model.eval()

        labels = ["N", "D", "G", "C", "A", "H", "M", "O"]

        count_N = 0
        count_D = 0
        count_else = 0

        files_DR = []
        files_DME = []
        files_N = []

        rootdir = '/scratch_net/petzi/garciaal/git/CANet/data'

        count_correct = 0
        count_total = 0
        count_wrongN = 0
        count_wrongD = 0

        model.cuda()

        list_subfolders_with_paths = [f.path for f in os.scandir(rootdir) if f.is_dir()]

        for dir in list_subfolders_with_paths:
            if "Base" not in dir:
                continue
            base_name = dir.split("/")[-1]
            print("Doing subdir ", base_name)
            directory = "/scratch_net/petzi/garciaal/git/CANet/data/" + base_name

            xl_workbook = xlrd.open_workbook(
                '/scratch_net/petzi/garciaal/git/CANet/data/' + base_name + '/Annotation ' + base_name + '.xls')
            xl_sheet = xl_workbook.sheet_by_index(0)
            for rowx in range(1, xl_sheet.nrows):
                cols = xl_sheet.row_values(rowx)
                filename = cols[0]
                label1 = int(cols[2])
                label2 = int(cols[3])

                if label1 > 0:
                    files_DR.append(filename)
                if label2 > 0:
                    files_DME.append(filename)
                if label1 == 0 and label2 == 0:
                    files_N.append(filename)

            with torch.no_grad():
                for filename in os.listdir(directory):
                    if filename.endswith(".tif") or filename.endswith(".png"):
                        path = os.path.join(directory, filename)
                        img = Image.open(path)
                        img = img.convert('RGB')

                        img = tra(img)
                        output = model(img.unsqueeze(0).cuda())
                        # pdb.set_trace()
                        result = torch.sigmoid(output)
                        if result[0][1] > 0.5:
                            final_result = labels[1]
                        else:
                            final_result = labels[0]

                        # final_result = result > 0.5
                        # all_false = not torch.any(final_result)
                        # pdb.set_trace()
                        # final_result = labels[torch.argmax(final_result).int()]
                        # pdb.set_trace()
                        # if all_false:
                        #     idx = torch.argmax(result)
                        #     final_result = labels[idx]

                        if final_result == "N":
                            count_N += 1
                        # elif  final_result == "O":
                        # count_O += 1
                        elif final_result == "D":
                            count_D += 1
                        else:
                            count_else += 1

                        if final_result == "N" and filename in files_N:
                            count_correct += 1
                        if final_result == "D" and filename in files_DR:
                            count_correct += 1

                        if final_result == "D" and filename in files_N:
                            count_wrongD += 1
                        if final_result == "N" and filename in files_DR:
                            count_wrongN += 1
                            # print("Processing image ", filename, " got result ", final_result)
                        # print("Processing image ", filename, " got result ", final_result)
                        count_total += 1
                    else:
                        continue
        # pdb.set_trace()
        print("Pseudo accuracy: ", count_correct * 1.0 / count_total)
        print("Wrong(is DR but classifies as N:", count_wrongN)
        print("Wrong(is N but classifies as DR:", count_wrongD)
        print("Normal: ", count_N, " DR: ", count_D)
        return

    if args.evaluate:
        a = time.time()
        # savedir = args.resume.replace("model_converge.pth.tar","")
        savedir = args.resume.replace(args.resume.split("/")[-1], "")
        # savedir = "./"
        if args.dataset == "ODIR" or args.dataset == "ODIR_EXT":
            precision, recall, accuracy, f1, val_loss, auc, acc_healthy, acc_dr = validate(val_loader, model, args)
            result_list = [precision, recall, accuracy, f1, val_loss, auc, acc_healthy, acc_dr]
            save_result_txt(savedir, result_list)
            print("time", time.time() - a)
            return
        elif not args.multitask:
            acc, auc, precision_dr, recall_dr, f1score_dr = validate(val_loader, model, args)
            result_list = [acc, auc, precision_dr, recall_dr, f1score_dr]
            print("acc, auc, precision, recall, f1", acc, auc, precision_dr, recall_dr, f1score_dr)
            save_result_txt(savedir, result_list)
            print("time", time.time() - a)
            return

        else:
            # print('validate?')
            acc_dr, acc_dme, acc_joint, other_results, se, sp = validate(val_loader, model, args)

            print("acc_dr, acc_dme, acc_joint", acc_dr, acc_dme, acc_joint)
            print("auc_dr, auc_dme, precision_dr, precision_dme, recall_dr, recall_dme, f1score_dr, f1score_dme",
                  other_results)
            print("se, sp", se, sp)
            result_list = [acc_dr, acc_dme, acc_joint]
            result_list += other_results
            result_list += [se, sp]
            save_result_txt(savedir, result_list)

            print("time", time.time() - a)
            return

    writer = SummaryWriter("runs/" + args.model_dir.split("/")[-1], flush_secs=1)
    writer.add_text('Text', str(args))
    #
    from lr_scheduler import LRScheduler
    lr_scheduler = LRScheduler(optimizer, len(train_loader), args)
    print("Epochs: ", args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        print("starting epoch: ", epoch)
        is_best = False
        is_best_auc = False
        is_best_acc = False
        # train for one epoch
        loss_train, train_acc = train(train_loader, model, criterion, lr_scheduler, writer, epoch, optimizer, args)
        writer.add_scalar('Train loss', loss_train, epoch)
        writer.add_scalar('Trainings accuracy', train_acc, epoch)

        # evaluate on validation set
        # TODO reset number to 20
        if epoch % 1 == 0:
            if args.dataset == "kaggle":
                acc_dr, auc_dr = validate(val_loader, model, args)
                writer.add_scalar("Val acc_dr", acc_dr, epoch)
                writer.add_scalar("Val auc_dr", auc_dr, epoch)
                best_acc1 = max(acc_dr, best_acc1)
                print("epoch {0}: accuracy: {1}\t auc: {2}".format(epoch, acc_dr, auc_dr))
            elif args.dataset == "ODIR" or args.dataset == "ODIR_EXT":
                [precision, recall, accuracy, f1, val_loss, auc, acc_healthy, acc_dr, acc_G, acc_C, acc_A, acc_H,
                 acc_M, acc_O, f1_healthy, f1_dr, f1_G, f1_C, f1_A, f1_H, f1_M, f1_O, auc_healthy, auc_dr, auc_G, auc_C,
                 auc_A, auc_H, auc_M, auc_O] = validate(val_loader, model, args)

                # Tensorboard every metric, careful the samples come from the validation loader
                writer.add_scalar("Val acc", accuracy, epoch)
                writer.add_scalar("Val auc", auc, epoch)
                writer.add_scalar('Val loss', val_loss, epoch)
                writer.add_scalar("Val f1", f1, epoch)
                writer.add_scalar("Acc Healthy", acc_healthy, epoch)
                writer.add_scalar("Acc DR", acc_dr, epoch)
                writer.add_scalar('Acc G', acc_G, epoch)
                writer.add_scalar('Acc C', acc_C, epoch)
                writer.add_scalar('Acc A', acc_A, epoch)
                writer.add_scalar('Acc H', acc_H, epoch)
                writer.add_scalar('Acc M', acc_M, epoch)
                writer.add_scalar('Acc O', acc_O, epoch)
                writer.add_scalar('f1 Healthy', f1_healthy, epoch)
                writer.add_scalar('f1 DR', f1_dr, epoch)
                writer.add_scalar('F1 G ', f1_G, epoch)
                writer.add_scalar('F1 C', f1_C, epoch)
                writer.add_scalar('F1 A', f1_A, epoch)
                writer.add_scalar('F1 H', f1_H, epoch)
                writer.add_scalar('F1 M', f1_M, epoch)
                writer.add_scalar('F1 O', f1_O, epoch)
                writer.add_scalar('AUC Healthy ', auc_healthy, epoch)
                writer.add_scalar('AUC DR', auc_dr, epoch)
                writer.add_scalar('AUC G', auc_G, epoch)
                writer.add_scalar('AUC C', auc_C, epoch)
                writer.add_scalar('AUC A', auc_A, epoch)
                writer.add_scalar('AUC H', auc_H, epoch)
                writer.add_scalar('AUC M', auc_M, epoch)
                writer.add_scalar('AUC O', auc_O, epoch)

                # choosing when to save checkpoint, if either accuracy !or! auc improves save checkpoint
                is_best_acc = accuracy >= best_acc1
                best_acc1 = max(best_acc1, accuracy)
                is_best_auc = auc >= best_auc
                best_auc = max(best_auc, auc)

                print("epoch {0}: accuracy: {1}\t f1-score: {2}\t auc-score: {3}".format(epoch, accuracy, f1, auc))
                save_checkpoint(epoch, model, optimizer, is_best_acc or is_best_auc,
                                filename='checkpoint-ep%d.pth.tar' % epoch,
                                save_dir=args.model_dir)
            elif not args.multitask:
                acc, auc, precision, recall, f1 = validate(val_loader, model, args)
                writer.add_scalar("Val acc_dr", acc, epoch)
                writer.add_scalar("Val auc_dr", auc, epoch)
                is_best = auc >= best_acc1
                best_acc1 = max(auc, best_acc1)
                print("epoch {0}: accuracy: {1}\t auc: {2}".format(epoch, acc, auc))
            else:
                acc_dr, acc_dme, joint_acc, other_results, se, sp = validate(val_loader, model, args)
                writer.add_scalar("Val acc_dr", acc_dr, epoch)
                writer.add_scalar("Val acc_dme", acc_dme, epoch)
                writer.add_scalar("Val acc_joint", joint_acc, epoch)
                writer.add_scalar("Val auc_dr", other_results[0], epoch)
                writer.add_scalar("Val auc_dme", other_results[1], epoch)
                is_best = joint_acc >= best_acc1
                best_acc1 = max(joint_acc, best_acc1)

                is_best_auc = other_results[0] >= best_auc
                best_auc = max(other_results[0], best_auc)

                is_best_acc = acc_dr >= best_accdr
                best_accdr = max(acc_dr, best_accdr)
                print("epoch {0}: accuracy: {1}\t acc_dme: {2}".format(epoch, acc_dr, acc_dme))
                save_checkpoint(epoch, model, optimizer, is_best_acc, filename='checkpoint.pth.tar', save_dir='file')
    # writer.close()


def train(train_loader, model, criterion, lr_scheduler, writer, epoch, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = 0
    # switch to train mode
    model.train()
    end = time.time()
    # print(len(train_loader))
    for i, (input, target, name, dataset) in enumerate(train_loader):
        target = target.long()

        # measure data loading time
        data_time.update(time.time() - end)
        lr = lr_scheduler.update(i, epoch)
        writer.add_scalar("lr", lr, epoch)

        if args.gpu is not None:
            # print("Init input")
            input = input.cuda(args.gpu, non_blocking=True)
            # print("Input is: ", input)

        if args.multitask:
            target = [item.cuda(args.gpu, non_blocking=True) for item in target]
        else:
            target = [item.cuda(args.gpu, non_blocking=True) for item in target]
            # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        if args.multitask:
            loss1 = criterion(output[0], target[0])
            loss2 = criterion(output[1], target[1])
            if args.crossCBAM:
                loss3 = criterion(output[2], target[0])
                loss4 = criterion(output[3], target[1])
                loss = (loss1 + loss2 + args.lambda_value * loss3 + args.lambda_value * loss4)
            else:
                loss = (loss1 + loss2)
        else:
            criterion_ml = nn.MultiLabelSoftMarginLoss(weight=class_weights, reduction='none').cuda(args.gpu)

            loss = criterion_ml(output, torch.stack(target).float())
            # Weight of the messidor dataset
            weight_m= torch.ones(loss.size(0)).to(loss.device)
            weight_m[np.array(dataset) == 'messidor'] *= 1
            loss = loss * weight_m
            loss = loss.mean()

            m = torch.sigmoid(output)
            m = np.uint8(m.detach().cpu() > 0.5)
            train_acc = accuracy_score(torch.stack(target).cpu(), m)

        losses.update(loss.item(), input.size(0))
        # print(losses.avg)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, train_acc


from sklearn.metrics import confusion_matrix


def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    all_target = []
    all_target_dme = []
    all_output = []
    all_name = []
    all_output_dme = []

    with torch.no_grad():
        for i, (input, target, name, dataset) in enumerate(val_loader):
            # print(i)
            if args.gpu is not None:
                # print("args.gpu not None")
                input = input.cuda(args.gpu, non_blocking=True)
            if args.multitask:
                # print("Multitask")
                target = [item.cuda(args.gpu, non_blocking=True) for item in target]
            else:
                # print("Not Multitask")
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)
            # pdb.set_trace()

            torch.cuda.synchronize()

            if args.dataset == "ODIR" or args.dataset == "ODIR_EXT":
                # all_target = target
                all_target.append(target)

                # pdb.set_trace()
                # all_output = output
                all_output.append(output)

            elif args.multitask:
                output0 = output[0]
                output1 = output[1]
                output0 = torch.softmax(output0, dim=1)
                output1 = torch.softmax(output1, dim=1)

                all_target.append(target[0].cpu().data.numpy())
                all_output.append(output0.cpu().data.numpy())
                all_target_dme.append(target[1].cpu().data.numpy())
                all_output_dme.append(output1.cpu().data.numpy())
            else:
                output = torch.softmax(output, dim=1)
                all_target.append(torch.Tensor(target.cpu().data.numpy()))
                all_output.append(output.cpu().data.numpy())
            all_name.append(name)

    if args.dataset == "kaggle":

        all_output = [item for sublist in all_output for item in sublist]
        all_target = [item for sublist in all_target for item in sublist]
        acc_dr = accuracy_score(all_target, np.argmax(all_output, axis=1))
        auc_dr = multi_class_auc(all_target, all_output, num_c=5)

        return acc_dr, auc_dr
    elif args.dataset == "ODIR" or args.dataset == "ODIR_EXT":

        # all_output = [item for sublist in all_output for item in sublist]
        # all_target = [item for sublist in all_target for item in sublist]

        # all_output = np.array(all_output)
        # all_target = np.array(all_target)

        # Loss function
        loss_fn = nn.MultiLabelSoftMarginLoss().cuda(args.gpu)

        val_loss = loss_fn(torch.cat(all_output).double(), torch.cat(all_target)).float()

        t_output = torch.cat(all_output)

        m = torch.sigmoid(t_output)
        m = m > 0.5
        m = m.cpu()

        target_tens = torch.cat(all_target).cpu()

        # Accuracy Metric for all 8 classes independently
        acc_healthy = accuracy_score(target_tens[:, 0], m[:, 0])
        acc_dr = accuracy_score(target_tens[:, 1], m[:, 1])
        acc_G = accuracy_score(target_tens[:, 2], m[:, 2])
        acc_C = accuracy_score(target_tens[:, 3], m[:, 3])
        acc_A = accuracy_score(target_tens[:, 4], m[:, 4])
        acc_H = accuracy_score(target_tens[:, 5], m[:, 5])
        acc_M = accuracy_score(target_tens[:, 6], m[:, 6])
        acc_O = accuracy_score(target_tens[:, 7], m[:, 7])

        # F1-Score for all 8 classes independently
        f1_healthy = f1_score(target_tens[:, 0], m[:, 0], average='binary')
        f1_dr = f1_score(target_tens[:, 1], m[:, 1], average='binary')
        f1_G = f1_score(target_tens[:, 2], m[:, 2], average='binary')
        f1_C = f1_score(target_tens[:, 3], m[:, 3], average='binary')
        f1_A = f1_score(target_tens[:, 4], m[:, 4], average='binary')
        f1_H = f1_score(target_tens[:, 5], m[:, 5], average='binary')
        f1_M = f1_score(target_tens[:, 6], m[:, 6], average='binary')
        f1_O = f1_score(target_tens[:, 7], m[:, 7], average='binary')

        # AUC for all 8 classes independently
        auc_healthy = roc_auc_score(target_tens[:, 0], m[:, 0])
        auc_dr = roc_auc_score(target_tens[:, 1], m[:, 1])
        auc_G = roc_auc_score(target_tens[:, 2], m[:, 2])
        auc_C = roc_auc_score(target_tens[:, 3], m[:, 3])
        auc_A = roc_auc_score(target_tens[:, 4], m[:, 4])
        auc_H = roc_auc_score(target_tens[:, 5], m[:, 5])
        auc_M = roc_auc_score(target_tens[:, 6], m[:, 6])
        auc_O = roc_auc_score(target_tens[:, 7], m[:, 7])

        # AUC score for whole Dataset
        auc = multi_class_auc(target_tens, m, num_c=8)

        precision, recall, accuracy, f1 = calc_acc_pr_f1_overall(target_tens, m)
        print("joint acc: %f\t healthy acc: %f\t dr acc: %f" % (accuracy, acc_healthy, acc_dr))

        # acc = accuracy_score(all_target, np.argmax(all_output, axis=1))
        # auc = multi_class_auc(all_target, all_output, num_c=8)
        # precision = precision_score(all_target, np.argmax(all_output, axis=1))
        return [precision, recall, accuracy, f1, val_loss, auc, acc_healthy, acc_dr, acc_G, acc_C, acc_A, acc_H, acc_M,
                acc_O, f1_healthy, f1_dr, f1_G, f1_C, f1_A, f1_H, f1_M, f1_O, auc_healthy, auc_dr, auc_G, auc_C, auc_A,
                auc_H, auc_M, auc_O]

    elif not args.multitask:
        all_target = [item for sublist in all_target for item in sublist]
        all_output = [item for sublist in all_output for item in sublist]

        if args.num_class == 2:
            acc = accuracy_score(all_target, np.argmax(all_output, axis=1))
            auc = roc_auc_score(all_target, [item[1] for item in all_output])
            precision_dr = precision_score(all_target, np.argmax(all_output, axis=1))
            recall_dr = recall_score(all_target, np.argmax(all_output, axis=1))
            f1score_dr = f1_score(all_target, np.argmax(all_output, axis=1))
        else:
            acc = accuracy_score(all_target, np.argmax(all_output, axis=1))
            auc = multi_class_auc(all_target, all_output, num_c=3)
            precision_dr = precision_score(all_target, np.argmax(all_output, axis=1), average="macro")
            recall_dr = recall_score(all_target, np.argmax(all_output, axis=1), average="macro")
            f1score_dr = f1_score(all_target, np.argmax(all_output, axis=1), average="macro")
        return acc, auc, precision_dr, recall_dr, f1score_dr

    else:
        all_target = [item for sublist in all_target for item in sublist]
        all_output = [item for sublist in all_output for item in sublist]
        all_target_dme = [item for sublist in all_target_dme for item in sublist]
        all_output_dme = [item for sublist in all_output_dme for item in sublist]

        # acc
        acc_dr = accuracy_score(all_target, np.argmax(all_output, axis=1))
        acc_dme = accuracy_score(all_target_dme, np.argmax(all_output_dme, axis=1))

        # joint acc
        joint_result = np.vstack((np.argmax(all_output, axis=1), np.argmax(all_output_dme, axis=1)))
        joint_target = np.vstack((all_target, all_target_dme))
        joint_acc = ((np.equal(joint_result, joint_target) == True).sum(axis=0) == 2).sum() / joint_result.shape[1]

        # auc
        if args.dataset == "missidor":
            auc_dr = roc_auc_score(all_target, [item[1] for item in all_output])
        else:
            auc_dr = multi_class_auc(all_target, all_output, num_c=5)
        auc_dme = multi_class_auc(all_target_dme, all_output_dme, num_c=3)

        # precision
        if args.dataset == "missidor":
            precision_dr = precision_score(all_target, np.argmax(all_output, axis=1))
        else:
            precision_dr = precision_score(all_target, np.argmax(all_output, axis=1), average="macro")
        precision_dme = precision_score(all_target_dme, np.argmax(all_output_dme, axis=1), average="macro")

        # recall
        if args.dataset == "missidor":
            recall_dr = recall_score(all_target, np.argmax(all_output, axis=1))
        else:
            recall_dr = recall_score(all_target, np.argmax(all_output, axis=1), average="macro")
        recall_dme = recall_score(all_target_dme, np.argmax(all_output_dme, axis=1), average="macro")
        print('f1score')
        # f1_score
        if args.dataset == "missidor":
            f1score_dr = f1_score(all_target, np.argmax(all_output, axis=1))
        else:
            f1score_dr = f1_score(all_target, np.argmax(all_output, axis=1), average="macro")
        f1score_dme = f1_score(all_target_dme, np.argmax(all_output_dme, axis=1), average="macro")

        cm1 = confusion_matrix(all_target, np.argmax(all_output, axis=1))
        sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

        return acc_dr, acc_dme, joint_acc, \
               [auc_dr, auc_dme, precision_dr, precision_dme, recall_dr, recall_dme, f1score_dr, f1score_dme], \
               sensitivity1, specificity1


def calc_acc_pr_f1_overall(y_true, y_pred):
    true = y_true
    pred = y_pred

    true[true == -1.] = 0

    precision = precision_score(true, pred, average='micro')
    recall = recall_score(true, pred, average='micro')

    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='micro')

    return precision, recall, accuracy, f1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def save_checkpoint(epoch, model, optimizer, is_best, filename='checkpoint.pth.tar', save_dir='file'):
    if not is_best:
        return

    root = save_dir + "/"
    if not os.path.exists(root):
        os.makedirs(root)
    # torch.save(state, root + filename)

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_dir + "/" + filename)


def save_result2txt(savedir, all_output_dme, all_output, all_target_dme, all_target):
    np.savetxt(savedir + "/output_dme.txt", all_output_dme, fmt='%.4f')
    np.savetxt(savedir + "/output_dr.txt", all_output, fmt='%.4f')
    np.savetxt(savedir + "/target_dme.txt", all_target_dme)
    np.savetxt(savedir + "/target_dr.txt", all_target)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def multi_class_auc(all_target, all_output, num_c=None):
    all_output = np.stack(all_output)
    all_target = label_binarize(all_target, classes=list(range(0, num_c)))
    auc_sum = []
    # print("multi_class_auc got target ", all_target)

    for num_class in range(0, num_c):
        try:
            auc = roc_auc_score(all_target[:, num_class], all_output[:, num_class])
            auc_sum.append(auc)
        except ValueError:
            print("Got value error!!")
            pass

    if len(auc_sum) == 0:
        auc_sum.append(0)

    auc = sum(auc_sum) / float(len(auc_sum))

    return auc


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels].cuda()


def save_result_txt(savedir, result):
    with open(savedir + '/result.txt', 'w') as f:
        for item in result:
            f.write("%.8f\n" % item)
        f.close()


if __name__ == '__main__':
    main()
