import argparse
import os

import torch
import numpy as np
from timm.data import ImageDataset, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from sklearn.metrics import roc_auc_score

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_transform(is_train, size):
    t = []
    t.append(transforms.Resize((size, size),
                                interpolation=InterpolationMode.BICUBIC, ))  # to maintain same ratio w.r.t. 224 images
    t.append(transforms.CenterCrop(size))
    if is_train:
        pass
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )   
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="Path to dataset",
    )  
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight Decay.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--input_size", default=224, type=int, help="input size.")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size.")
    
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    
    args = parser.parse_args()
    return args


def main(args):
    
    margin = 1.0
    imratio = 0.5
    
    train_transforms = build_transform(is_train=True, size=args.input_size)
    val_transforms = build_transform(is_train=False, size=args.input_size)
    class_map = {'non_monkeypox': 0, 'monkeypox': 1}
    
    train_dataset = ImageDataset(os.path.join(args.dataset_path , "train"), transform=train_transforms,
                                     class_map=class_map)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=args.batch_size,
                                num_workers=1,
                                pin_memory=True,
                                persistent_workers=True,
                                drop_last=True)
    
    val_dataset = ImageDataset(os.path.join(args.dataset_path , "val"), transform=val_transforms,
                                     class_map=class_map)
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset,
                                sampler=val_sampler,
                                batch_size=args.batch_size,
                                num_workers=1,
                                pin_memory=True,
                                persistent_workers=True,
                                drop_last=False)
    
    set_all_seeds(args.seed)
    model = DenseNet121(pretrained=False, last_activation=None, activations='relu', num_classes=1)
    model = model.cuda()
    
    if True:
        PATH = args.pretrained_model_path
        state_dict = torch.load(PATH)
        state_dict.pop('classifier.weight', None)
        state_dict.pop('classifier.bias', None) 
        model.load_state_dict(state_dict, strict=False)
    
    loss_fn = AUCMLoss(imratio=imratio)
    optimizer = PESG(model, 
                    loss_fn=loss_fn, 
                    lr=args.learning_rate, 
                    margin=margin, 
                    epoch_decay=2e-3, 
                    weight_decay=args.weight_decay)
    
    best_val_auc = 0
    for epoch in range(5):
        if epoch > 0:
            optimizer.update_regularizer(decay_factor=2)
        for idx, data in enumerate(train_loader):
            train_data, train_labels = data
            train_labels = train_labels.type(torch.FloatTensor)
            train_data, train_labels = train_data.cuda(), train_labels.cuda()
            y_pred = model(train_data)
            y_pred = torch.sigmoid(y_pred)
            loss = loss_fn(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation
            if idx % 10 == 0:
                model.eval()
                with torch.no_grad():    
                    test_pred = []
                    test_true = [] 
                    for jdx, data in enumerate(val_loader):
                        test_data, test_label = data
                        test_label = test_label.type(torch.FloatTensor)
                        test_data = test_data.cuda()
                        y_pred = model(test_data)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_label.numpy())
                    
                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc =  roc_auc_score(test_true, test_pred) 
                    model.train()

                    if best_val_auc < val_auc:
                        best_val_auc = val_auc
                        torch.save(model.state_dict(), 'finetuned_model.pth')
                    
                print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, lr=%.4f'%(epoch, idx, val_auc, optimizer.lr))

    print ('Best Val_AUC is %.4f'%best_val_auc)
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
