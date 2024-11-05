import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, nb_channel=64):
        super(UNet, self).__init__()

        # Downward path (encoder)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nb_channel, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel, nb_channel, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(nb_channel, nb_channel * 2, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel * 2, nb_channel * 2, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(nb_channel * 2, nb_channel * 4, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel * 4, nb_channel * 4, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(nb_channel * 4, nb_channel * 8, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel * 8, nb_channel * 8, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv5 = nn.Sequential(
            nn.Conv2d(nb_channel * 8, nb_channel * 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel * 16, nb_channel * 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.drop5 = nn.Dropout(0.5)

        # Upward path (decoder)
        self.up6 = nn.ConvTranspose2d(nb_channel * 16, nb_channel * 8, kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Sequential(
            nn.Conv2d(nb_channel * 16, nb_channel * 8, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel * 8, nb_channel * 8, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )

        self.up7 = nn.ConvTranspose2d(nb_channel * 8, nb_channel * 4, kernel_size=2, stride=2, padding=0)
        self.conv7 = nn.Sequential(
            nn.Conv2d(nb_channel * 8, nb_channel * 4, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel * 4, nb_channel * 4, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )

        self.up8 = nn.ConvTranspose2d(nb_channel * 4, nb_channel * 2, kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Sequential(
            nn.Conv2d(nb_channel * 4, nb_channel * 2, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel * 2, nb_channel * 2, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )

        self.up9 = nn.ConvTranspose2d(nb_channel * 2, nb_channel, kernel_size=2, stride=2, padding=0)
        self.conv9 = nn.Sequential(
            nn.Conv2d(nb_channel * 2, nb_channel, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channel, nb_channel, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.conv10 = nn.Conv2d(nb_channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        # Decoder
        up6 = self.up6(drop5)
        merge6 = torch.cat((drop4, up6), dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat((conv3, up7), dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat((conv2, up8), dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat((conv1, up9), dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)
        output = self.sigmoid(conv10)

        return output
