import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn import Linear, functional as F, Dropout
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from Losses.MeanVarianceLoss import MeanVarianceLoss
from GradualWarmupScheduler import GradualWarmupScheduler

from Training.train_unified_model_iter import train_unified_model_iter


TRAIN_CSV_PATH = './UTKFace/utkface_train.csv'
TEST_CSV_PATH = './UTKFace/utkface_test.csv'
IMAGE_PATH = './UTKFace/jpg'

seed = 3407
torch.manual_seed(seed)

np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
num_iters = int(6e4)

# Hyperparameters
learning_rate = 0.0005
num_epochs = 200

# Architecture
NUM_CLASSES = 14
BATCH_SIZE = 32
# GRAYSCALE = False
min_age = 21
max_age = 60
age_interval = 3

df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
ages = df['age'].values
del df
ages = torch.tensor(ages, dtype=torch.float)


class UTKFaceDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]//3  # interval=3
        levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        sample = {'image': img, 'classification_label': label,
                  'age': self.y[index]+min_age}
        return sample

    def __len__(self):
        return self.y.shape[0]


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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5)
])

train_dataset = UTKFaceDataset(csv_path=TRAIN_CSV_PATH,
                               img_dir=IMAGE_PATH,
                               transform=custom_transform)

custom_transform2 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = UTKFaceDataset(csv_path=TEST_CSV_PATH,
                              img_dir=IMAGE_PATH,
                              transform=custom_transform2)

if __name__ == '__main__':

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
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

    class FeatureExtractionswintransformer(nn.Module):
        def __init__(self):
            super(FeatureExtractionswintransformer, self).__init__()

            model = models.swin_t(pretrained=True)
            all_layers = list(model.children())

            new_model = torch.nn.Sequential(*all_layers[:-1])
            self.base_net = new_model

        def forward(self, input):
            x = self.base_net(input)

            return x

    class UnifiedClassificaionAndRegressionAgeModel(nn.Module):
        def __init__(self, num_classes, age_interval, min_age, max_age, device=torch.device("cuda:0")):
            super(UnifiedClassificaionAndRegressionAgeModel, self).__init__()
            self.device = device
            self.age_intareval = age_interval
            self.min_age = min_age
            self.max_age = max_age
            self.labels = range(num_classes)

            self.base_net = FeatureExtractionswintransformer()
            k = 768
            self.num_features = 768

            self.fc_first_stage = Linear(self.num_features, k)
            self.class_head = Linear(k, num_classes)

            self.fc_second_stage = Linear(self.num_features, k)

            self.regression_heads = []
            self.centers = []
            for i in self.labels:
                self.regression_heads.append(Linear(k, 1))
                # self.classification_heads.append(Linear(k, int(self.age_intareval*2)))
                center = min_age + 0.5 * age_interval + i * age_interval
                self.centers.append(center)

            self.regression_heads = nn.ModuleList(self.regression_heads)

        def freeze_base_cnn(self, should_freeze=True):
            for param in self.base_net.parameters():
                param.requires_grad = not should_freeze

        def forward(self, input_images, p=0.5):

            base_embedding = self.base_net(input_images)
            base_embedding = F.leaky_relu(base_embedding)
            # base_embedding = Dropout(p)(base_embedding)

            # first stage
            class_embed = self.fc_first_stage(base_embedding)
            class_embed = F.normalize(class_embed, dim=1, p=2)
            x = F.leaky_relu(class_embed)
            x = Dropout(p)(x)

            classification_logits = self.class_head(x)

            weights = nn.Softmax()(classification_logits)

            # second stage
            x = self.fc_second_stage(base_embedding)
            x = F.leaky_relu(x)
            x = Dropout(p)(x)

            t = []
            for i in self.labels:
                t.append(torch.squeeze(self.regression_heads[i](
                    x) + self.centers[i]) * weights[:, i])

            age_pred = torch.stack(t, dim=0).sum(
                dim=0) / torch.sum(weights, dim=1)

            return classification_logits, age_pred
        # create model and parameters

    model = UnifiedClassificaionAndRegressionAgeModel(
        NUM_CLASSES, age_interval, min_age, max_age)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    criterion_reg = nn.MSELoss().to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    mean_var_criterion = MeanVarianceLoss(
        0, NUM_CLASSES, device, lambda_mean=0.2, lambda_variance=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # num_epochs = int(num_iters / len(train_loader) + 1)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_iters
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=8000,
        after_scheduler=cosine_scheduler
    )
    # train model
    # print(model)
    writer = SummaryWriter(
        'logs/utk/swintransformer-bs32-bin3')
    # writer.add_image('images[0]', images[0])
    # writer.add_graph(model, images[0].unsqueeze(0))

    model_path = 'weights/swintransformer-bs32-bin3'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

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
        NUM_CLASSES,
        num_epochs=num_epochs,
        validate_at_k=500
    )

    FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
    torch.save(best_model.state_dict(), FINAL_MODEL_FILE)

    print('fun fun in the sun, the training is done :)')
