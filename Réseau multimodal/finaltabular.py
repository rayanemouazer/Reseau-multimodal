import numpy as np
import os
import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from scipy import ndimage as ndi
from PIL import Image
from skimage.segmentation import mark_boundaries
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Resize
)
from sklearn.model_selection import ShuffleSplit
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torchvision.transforms import ToTensor
from skimage.measure import label, regionprops
from sklearn.preprocessing import StandardScaler


# Chemin du dossier d'entraînement
path2train = "./data/images2/"

# Liste des images et des annotations
imgsList = [pp for pp in os.listdir(path2train) if "Annotation" not in pp]
anntsList = [pp for pp in os.listdir(path2train) if "Annotation" in pp]

# Affichage du nombre d'images et d'annotations
print("nombre d'images :", len(imgsList))
print("nombre d'annotations :", len(anntsList))

# Sélection aléatoire de 4 images
np.random.seed(2019)
rndImgs = np.random.choice(imgsList, 4)
print(rndImgs)


def show_img_mask(img, mask):
    img_np = np.array(to_pil_image(img))
    mask_np = np.array(to_pil_image(mask))

    # Assurez-vous que les dimensions de mask_np correspondent à celles de img_np
    mask_np_resized = np.resize(mask_np, img_np.shape[:2])

    # Utilisez mark_boundaries avec img_np et mask_np_resized
    img_mask = mark_boundaries(img_np.astype(np.uint8), mask_np_resized.astype(np.uint8), outline_color=(0, 1, 0), color=(0, 1, 0))
    plt.imshow(img_mask)
    plt.axis('off')
    plt.show()


# Transformation pour l'entraînement
h, w = 128, 192
transform_train = Compose([
    Resize(h, w),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
])

# Transformation pour la validation
transform_val = Resize(h, w)


class fetal_dataset(Dataset):

    def __init__(self, path2data, tabular_data_path, transform=None):
        # Charger les chemins des images et annotations
        imgsList = [pp for pp in os.listdir(path2data) if "Annotation" not in pp and pp.endswith(('.png', '.jpg', '.jpeg'))]
        anntsList = [pp for pp in os.listdir(path2data) if "Annotation" in pp and pp.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.path2imgs = [os.path.join(path2data, fn) for fn in imgsList]
        self.path2annts = [p2i.replace(".png", "_Annotation.png") for p2i in self.path2imgs]
        
        # Charger les données tabulaires
        self.tabular_data = pd.read_csv(tabular_data_path)
        self.tabular_data.set_index('id', inplace=True)

        # Sélectionner les colonnes à normaliser
        colonnes_features = ['area_mean', 'perimeter_mean']  # Ajoutez ici toutes les colonnes que vous souhaitez normaliser
        self.tabular_data = self.tabular_data[colonnes_features]

        # Normaliser les colonnes spécifiées
        scaler = StandardScaler()
        self.tabular_data[colonnes_features] = scaler.fit_transform(self.tabular_data[colonnes_features])
        
        self.transform = transform

    def __len__(self):
        return len(self.path2imgs)

    def __getitem__(self, idx):
        # Charger l'image et le masque
        path2img = self.path2imgs[idx]
        image = Image.open(path2img)
        path2annt = self.path2annts[idx]
        annt_edges = Image.open(path2annt)
        mask = ndi.binary_fill_holes(np.array(annt_edges))
        
        image = np.array(image)
        mask = mask.astype("uint8")
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = to_tensor(image)
        mask = 255 * to_tensor(mask)
        
        # Charger les données tabulaires associées
        img_id = int(os.path.splitext(os.path.basename(path2img))[0])  # Supposons que l'ID de l'image est dans le nom de fichier
        tabular_features = self.tabular_data.loc[img_id].values.astype(np.float32)
        
        return image, mask, tabular_features


# Chemin du dossier d'entraînement
tabularpath = "./data/data.csv"

# Création des objets de la classe fetal_dataset
fetal_ds1 = fetal_dataset(path2train, tabularpath, transform=transform_train)
fetal_ds2 = fetal_dataset(path2train, tabularpath, transform=transform_val)

# Affichage du nombre d'éléments dans les ensembles de données
print(len(fetal_ds1))
print(len(fetal_ds2))

# Récupérer un élément du dataset
image, mask, tabular_features = fetal_ds1[0]

# Vérification des formes et des données
print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Tabular features: {tabular_features}")

# Display the image and the mask
show_img_mask(image, mask)

# Split the data
sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices = range(len(fetal_ds1))
for train_index, val_index in sss.split(indices):
    print(len(train_index))
    print("-" * 10)
    print(len(val_index))

# Create train_ds and val_ds:
train_ds = Subset(fetal_ds1, train_index)
print(len(train_ds))
val_ds = Subset(fetal_ds2, val_index)
print(len(val_ds))

# Show a simple image from train_ds
# Show a simple image from train_ds
plt.figure(figsize=(5, 5))
for img, mask, _ in train_ds:
    show_img_mask(img, mask)
    break

# Show a sample image and mask from val_ds:
plt.figure(figsize=(5, 5))
for img, mask, _ in val_ds:
    show_img_mask(img, mask)
    break

# Define the data loaders
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

# MODEL


class UNetWithTabular(nn.Module):

    def __init__(self, in_channels=1, num_classes=1, tabular_input_dim=2):
        super(UNetWithTabular, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.encoder_conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Branche tabulaire
        self.tabular_branch = nn.Sequential(
    nn.Linear(tabular_input_dim, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU()
)

        # Fusion
        self.fusion_layer = nn.Linear(1024 + 64, 1024)

    def forward(self, x, tabular_data):
        # Encoder
        x1 = torch.relu(self.encoder_conv1(x))
        x1 = torch.relu(self.encoder_conv2(x1))
        pool1 = self.pool1(x1)

        x2 = torch.relu(self.encoder_conv3(pool1))
        x2 = torch.relu(self.encoder_conv4(x2))
        pool2 = self.pool2(x2)

        x3 = torch.relu(self.encoder_conv5(pool2))
        x3 = torch.relu(self.encoder_conv6(x3))
        pool3 = self.pool3(x3)

        x4 = torch.relu(self.encoder_conv7(pool3))
        x4 = torch.relu(self.encoder_conv8(x4))
        pool4 = self.pool4(x4)

        x5 = torch.relu(self.encoder_conv9(pool4))
        x5 = torch.relu(self.encoder_conv10(x5))

        # Branche tabulaire
        tabular_features = self.tabular_branch(tabular_data)  # [batch_size, 64]
        tabular_features = tabular_features.expand(x5.size(0), -1)  # Ajuster la dimension de batch
        tabular_features = tabular_features.unsqueeze(2).unsqueeze(3)  # Redimensionner [batch_size, 64, 1, 1]
        tabular_features = tabular_features.expand(-1, -1, x5.size(2), x5.size(3))  # [batch_size, 64, height, width]

        
        print(f"x5 shape: {x5.shape}")  # Devrait être [batch_size, channels, height, width]
        print(f"tabular_features shape: {tabular_features.shape}")  # Devrait être [batch_size, -1]


        # Modifier fusion_layer
        self.fusion_layer = nn.Conv2d(1088, 1024, kernel_size=1)  # Fusion sur la dimension des canaux

        # Combiner sans aplatir
        combined = torch.cat((x5, tabular_features), dim=1)  # [batch_size, 1088, height, width]

        # Passer dans fusion_layer
        fused = self.fusion_layer(combined)  # [batch_size, 1024, height, width]

        # Vérifiez la taille de fused
        print(f"Fused shape (before reshaping): {fused.shape}")

        # Récupérer les dimensions dynamiquement
        batch_size, channels, height, width = fused.size()

        # Vérifiez les dimensions récupérées
        print(f"Batch size: {batch_size}, Channels: {channels}, Height: {height}, Width: {width}")




        # Decoder
        upconv1 = self.upconv1(fused)
        concat1 = torch.cat([upconv1, x4], dim=1)
        x6 = torch.relu(self.decoder_conv1(concat1))
        x6 = torch.relu(self.decoder_conv2(x6))

        upconv2 = self.upconv2(x6)
        concat2 = torch.cat([upconv2, x3], dim=1)
        x7 = torch.relu(self.decoder_conv3(concat2))
        x7 = torch.relu(self.decoder_conv4(x7))

        upconv3 = self.upconv3(x7)
        concat3 = torch.cat([upconv3, x2], dim=1)
        x8 = torch.relu(self.decoder_conv5(concat3))
        x8 = torch.relu(self.decoder_conv6(x8))

        upconv4 = self.upconv4(x8)
        concat4 = torch.cat([upconv4, x1], dim=1)
        x9 = torch.relu(self.decoder_conv7(concat4))
        x9 = torch.relu(self.decoder_conv8(x9))

        output = self.output_conv(x9)
        return output


class ModelForSummary(nn.Module):

    def __init__(self, model, tabular_features):
        super(ModelForSummary, self).__init__()
        self.model = model
        self.tabular_features = tabular_features

    def forward(self, x):
        # Passe les `tabular_features` au modèle dans le cadre de l'appel forward
        return self.model(x, self.tabular_features)


# Instancier le modèle et transférer sur le bon dispositif
model = UNetWithTabular()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = image.unsqueeze(0).to(device)  # Ajouter une dimension batch et transférer sur le dispositif
tabular_features = torch.tensor(tabular_features, dtype=torch.float32).unsqueeze(0).to(device)

model.to(device)

# Créer une version encapsulée du modèle
summary_model = ModelForSummary(model, tabular_features)

summary(summary_model, input_size=(1, *image.shape[2:]))

# Loss et optimizer

# Define a helper fonction


def show_img_mask_with_stats(img, mask, mean_size, num_cells):
    img_np = np.array(img)
    if img_np.ndim == 2:  # Si l'image est en niveaux de gris
        img_np = np.expand_dims(img_np, axis=-1)
        img_np = np.concatenate([img_np] * 3, axis=-1)
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    img_mask = mark_boundaries(img_np, mask_np, outline_color=(0, 1, 0), color=(0, 1, 0))
    
    plt.imshow(img_mask)
    plt.axis('off')
    plt.title(f'Mean Cell Size: {mean_size:.2f}, Number of Cells: {num_cells}')
    plt.show()


class EarlyStopping:

    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def IoU(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.sum()


def dice_score(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.sum()


def loss_func(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    pred = torch.sigmoid(pred)
    dlv = 1.0 - dice_score(pred, target)
    loss = bce + dlv
    return loss


def metrics_batch(pred, target):
    pred = torch.sigmoid(pred)
    _, metric = dice_loss(pred, target)
    return metric


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    pred = torch.sigmoid(output)
    metric_b = dice_score(pred, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b


opt = optim.Adam(model.parameters(), lr=3e-3)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=1)


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


current_lr = get_lr(opt)
print('current lr={}'.format(current_lr))

# Training the model


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    running_iou = 0.0
    len_data = len(dataset_dl.dataset)
    
    for xb, yb, tabular_data in dataset_dl:
        # Envoyer les données sur le bon périphérique
        xb = xb[:, 0:1,:,:].type(torch.float32).to(device)
        yb = yb.type(torch.float32).to(device)
        tabular_data = tabular_data.type(torch.float32).to(device)
        
        output = model(xb,tabular_data)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        if metric_b is not None:
            running_metric += metric_b
        iou_b = IoU(torch.sigmoid(output), yb)
        running_iou += iou_b
        if sanity_check:
            break
    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    iou = running_iou / float(len_data)
    return loss, metric, iou


def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    early_stopping = EarlyStopping(patience=20, verbose=True, path=path2weights)
    
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    iou_history = {"train": [], "val": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

        # Training phase
        model.train()
        train_loss, train_metric, train_iou = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        iou_history["train"].append(train_iou)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_metric, val_iou = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        iou_history["val"].append(val_iou)

        # Check if the current model is the best one
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Learning rate scheduling
        lr_scheduler.step(val_loss)

        # If learning rate changed, load the best model weights
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        # Print the training and validation metrics
        print("train loss: %.6f, dice: %.2f, IoU: %.2f" % (train_loss, 100 * train_metric, 100 * train_iou))
        print("val loss: %.6f, dice: %.2f, IoU: %.2f" % (val_loss, 100 * val_metric, 100 * val_iou))
        print("-" * 10)

    # Load the best model weights at the end of training
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history, iou_history


# Call train_val
# Initialisation de l'optimiseur et du scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

# Dossier pour enregistrer les poids du modèle
path2models = "./models/"
if not os.path.exists(path2models):
    os.mkdir(path2models)

# Définition des paramètres d'entraînement
params_train = {
    "num_epochs": 120,
    "optimizer": optimizer,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": scheduler,
    "path2weights": path2models + "weights.pt",
}

# Appel de la fonction train_val avec les paramètres d'entraînement
model, loss_hist, metric_hist, iou_hist = train_val(model, params_train)
# Number of epochs completed
num_epochs_completed = len(loss_hist["train"])

# Plot the progress of the training and validation losses
plt.figure()
plt.title("Train-Val Loss")
plt.plot(range(1, num_epochs_completed + 1), loss_hist["train"], label="train")
plt.plot(range(1, num_epochs_completed + 1), loss_hist["val"], label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# Plot the dice score
plt.figure()
plt.title("Train-Val Dice Score")
plt.plot(range(1, num_epochs_completed + 1), [m.detach().cpu().numpy() for m in metric_hist["train"]], label="train")
plt.plot(range(1, num_epochs_completed + 1), [m.detach().cpu().numpy() for m in metric_hist["val"]], label="val")
plt.ylabel("Dice Score")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# Plot the IoU
plt.figure()
plt.title("Train-Val IoU")
plt.plot(range(1, num_epochs_completed + 1), [m.detach().cpu().numpy() for m in iou_hist["train"]], label="train")
plt.plot(range(1, num_epochs_completed + 1), [m.detach().cpu().numpy() for m in iou_hist["val"]], label="val")
plt.ylabel("IoU")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()


# Fonction pour montrer l'image et le masque
def show_img_mask(img, mask):
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformation pour l'image
to_tensor = ToTensor()

# Chemin du dossier de test
path2test = "./data/images/"
imgsList = [pp for pp in os.listdir(path2test) if "Annotation" not in pp]
print("number of images:", len(imgsList))

# Sélection aléatoire de 4 images de test
np.random.seed(2019)
rndImgs = np.random.choice(imgsList, 4)
print(rndImgs)

# Chemin du fichier de poids
path2weights = "./models/weights.pt"

# Vérifiez l'espace disponible avant de sauvegarder
import shutil
total, used, free = shutil.disk_usage("/")

if free < 1:  # Moins de 1 GiB d'espace libre
    print("Not enough disk space to save the model.")
else:
    
    # Initialisation des listes pour stocker les résultats
    results = []

    # Définir les dimensions de redimensionnement (w et h)
    w, h = 256, 256  # ou les dimensions souhaitées

    # Affichage des images, masques prédits et combinaisons
    for fn in rndImgs:
        path2img = os.path.join(path2test, fn)
        
        try:
            with Image.open(path2img) as img_pil:
                img_pil = img_pil.convert('L')  # Conversion en niveaux de gris
                img = np.array(img_pil)  # Conversion en numpy array
        except Exception as e:
            print(f"Error loading image {fn}: {e}. Skipping this file.")
            continue
        
        if img is None:
            print(f"Error loading image {fn}. Skipping this file.")
            continue
        
        img = cv2.resize(img, (w, h))
        img_t = to_tensor(img).unsqueeze(0).to(device)
        
        try:
            pred = model(img_t,tabular_features)
            pred = torch.sigmoid(pred)[0]
            mask_pred = (pred[0] > 0.27).cpu().numpy().astype(np.uint8)  # Seuillage du masque prédictif et conversion en numpy uint8
        except Exception as e:
            print(f"Error during prediction for image {fn}: {e}")
            continue
        
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.title('Image originale')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_pred, cmap="gray")
        plt.axis('off')
        plt.title('Masque prédit')
        
        plt.subplot(1, 3, 3)
        show_img_mask(img, mask_pred)
        plt.title('Image avec masque')
        
        plt.show()

    # Sauvegarde du modèle avec gestion des exceptions
    try:
        torch.save(model.state_dict(), path2weights)
    except Exception as e:
        print(f"Error saving model weights: {e}")
    
