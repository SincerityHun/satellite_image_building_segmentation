from model import Unet_block, Nested_UNet
import torch
from tqdm import tqdm
from data_loader import *
from dataset import SatelliteDataset, DataLoader
import numpy as np
from utils import *

# 1. Add these at the beginning of the script
import pandas as pd
from joblib import Parallel, delayed
from typing import List

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS Used")
else:
    device = torch.device("cpu")
    print("CPU Used")

if __name__ == "__main__":
    # 1. Train
    # model 초기화
    model = Nested_UNet(1, 3, deep_supervision=False).to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000068)
    EPOCH = 1
    best_dice = 0

    # training loop
    for epoch in range(EPOCH):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        for images, masks in tqdm(train_loader):
            # 1. 이미지, 마스크 설정
            images = images.float().to(device)
            masks = masks.float().to(device)

            # 2. Optimizer 설정
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))

            outputs_np = outputs.detach().cpu().numpy()
            masks_np = masks.cpu().numpy()
            loss = criterion(outputs, masks.unsqueeze(1))
            dice = dice_coef(outputs, masks.unsqueeze(1))

            # 3. backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice.item()

        print(
            f"Training Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}, Dice: {epoch_dice/len(train_loader)}"
        )

        # validation
        model.eval()
        valid_loss = 0
        epoch_dice = 0

        with torch.no_grad():
            for images, masks in tqdm(valid_loader):
                images = images.float().to(device)
                masks = masks.float().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                dice = dice_coef(outputs, masks.unsqueeze(1))

                valid_loss += loss.item()
                epoch_dice += dice.item()
        valid_dice = epoch_dice / len(valid_loader)
        print(
            f"Valid Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}, Dice: {valid_dice}"
        )
        if best_dice < valid_dice:
            best_dice = valid_dice
            torch.save(model.state_dict(), f"./results/unet/best_model.pth")

    # # 4. Inference
    # with torch.no_grad():
    #     model.eval()
    #     result = []
    #     for images in tqdm(test_dataloader):
    #         images = images.float().to(device)

    #         outputs = model(images)
    #         masks = torch.sigmoid(outputs).cpu().numpy()
    #         masks = np.squeeze(masks, axis=1)
    #         masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35

    #         for i in range(len(images)):
    #             mask_rle = rle_encode(masks[i])
    #             if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
    #                 result.append(-1)
    #             else:
    #                 result.append(mask_rle)

    # # 4. Submit
    # submission('mask_rle',result)
