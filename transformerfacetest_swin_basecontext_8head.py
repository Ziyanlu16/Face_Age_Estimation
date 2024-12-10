from models.transformer import TransformerModel
from models.agetransformer import AgeTransformer
from models.cnn2 import UnifiedClassificaionAndRegressionAgeModel
from DataSet.UTKFaceDataset import UTKFaceDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
import random
import pandas as pd
import torchvision.transforms.v2 as transforms
from Losses.MeanVarianceLoss import MeanVarianceLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from GradualWarmupScheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
import os
from Training.train_unified_model_iter import train_unified_model_iter


def get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size):
    pretrained_model = UnifiedClassificaionAndRegressionAgeModel(
        num_classes, age_interval, min_age, max_age)
    # pretrained_model_path='weights/utk/cnn'
    # pretrained_model_path='weights/utk/resnet50' #只是一次测试，要改过来
    pretrained_model_file = 'weights/swintransformer-bs32-bin3/weights_10500_4.4411.pt'
    # pretrained_model_file = os.path.join(pretrained_model_path, "weights.pt")
    pretrained_model.load_state_dict(
        torch.load(pretrained_model_file), strict=False)

    num_features = pretrained_model.num_features
    backbone = pretrained_model.base_net
    backbone.train()
    backbone.to(device)

    transformer = TransformerModel(
        age_interval, min_age, max_age,
        mid_feature_size, mid_feature_size,
        num_outputs=num_classes,
        n_heads=8, n_encoders=4, dropout=0.3,
        mode='context').to(device)
    age_transformer = AgeTransformer(
        backbone, transformer, num_features, mid_feature_size).to(device)

    return age_transformer


if __name__ == "__main__":
    seed = 3407
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True
    min_age = 21
    max_age = 60
    age_interval = 3
    batch_size = 12
    num_iters = int(8e4)
    random_split = True
    num_copies = 10
    mid_feature_size = 1024

    num_classes = int((max_age - min_age) / age_interval + 1)
    # NUM_CLASSES = 40

    TRAIN_CSV_PATH = './UTKFace/utkface_train.csv'
    TEST_CSV_PATH = './UTKFace/utkface_test.csv'
    IMAGE_PATH = './UTKFace/jpg'
    df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
    ages = df['age'].values
    del df
    ages = torch.tensor(ages, dtype=torch.float)
    custom_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, (0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1
        )], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        )], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])
    train_dataset = UTKFaceDataset(csv_path=TRAIN_CSV_PATH,
                                   img_dir=IMAGE_PATH,
                                   transform=custom_transform, copies=num_copies)
    test_dataset = UTKFaceDataset(
        csv_path=TEST_CSV_PATH, img_dir=IMAGE_PATH, transform=custom_transform, copies=num_copies)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=4)
    # images=next(iter(train_loader))['image']
    image_datasets = {
        'train': train_dataset,
        'val': test_dataset
    }
    data_loaders = {
        'train': train_loader,
        'val': test_loader
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model = get_age_transformer(
        device, num_classes, age_interval, min_age, max_age, mid_feature_size)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    criterion_reg = nn.MSELoss().to(device)
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)

    mean_var_criterion = MeanVarianceLoss(
        0, num_classes, device, lambda_mean=0.2, lambda_variance=0.05).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5)
    # optimizer = RangerLars(model.parameters(), lr=1e-4)
    num_epochs = int(num_iters / len(data_loaders['train'])) + 1
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_iters
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=8000,
        # total_epoch=5,
        after_scheduler=cosine_scheduler
    )

    ### Train ###
    writer = SummaryWriter('logs/utk/transformer13-6head')
    # writer = None

    model_path = 'weights/transformer13-6head'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # model_path = NoneS

    best_model = train_unified_model_iter(
        model,
        criterion_reg,
        criterion_cls,
        mean_var_criterion,
        optimizer,
        scheduler,
        data_loaders,
        dataset_sizes,
        device,
        writer,
        model_path,
        num_classes,
        num_epochs=num_epochs,
        validate_at_k=900)

    print('saving best model')

    FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
    torch.save(best_model.state_dict(), FINAL_MODEL_FILE)

    print('fun fun in the sun, the training is done :)')
