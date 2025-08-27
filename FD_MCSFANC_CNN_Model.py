import torch
import torchvision.models as models
import torch.nn as nn


class Modified_ShufflenetV2_Frequency_DOA(nn.Module):
    def __init__(self, num_classes1=7, num_classes2=8):
        super().__init__()

        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2_base_conv1 = models.shufflenet_v2_x0_5(weights='DEFAULT').conv1
        self.mv2_base_maxpool = models.shufflenet_v2_x0_5(weights='DEFAULT').maxpool
        self.mv2_base_fre_stage2 = models.shufflenet_v2_x0_5(weights='DEFAULT').stage2
        self.mv2_base_fre_stage3 = models.shufflenet_v2_x0_5(weights='DEFAULT').stage3
        self.mv2_base_doa_stage2 = models.shufflenet_v2_x0_5(weights='DEFAULT').stage2
        self.mv2_base_doa_stage3 = models.shufflenet_v2_x0_5(weights='DEFAULT').stage3
        self.avg_pool_fre = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool_doa = nn.AdaptiveAvgPool2d((1, 1))

        # Task-specific layers for frequency classification
        self.freq_fc = nn.Linear(96, num_classes1)

        # Task-specific layers for DOA classification
        self.doa_fc = nn.Linear(96, num_classes2)

    def forward(self, x): # torch.Size([Batch, 8, 64, 126])
        x = self.bw2col(x) # torch.Size([Batch, 3, 64, 126])
        x = self.mv2_base_conv1(x) # torch.Size([Batch, 24, 32, 63])
        x = self.mv2_base_maxpool(x) # torch.Size([Batch, 24, 16, 32])
        # Frequency classification
        x1 = self.mv2_base_fre_stage2(x) # torch.Size([Batch, 48, 8, 16])
        x1 = self.mv2_base_fre_stage3(x1) # torch.Size([Batch, 96, 4, 8])
        x1 = self.avg_pool_fre(x1).squeeze(-1).squeeze(-1) # torch.Size([Batch, 96])
        out1 = self.freq_fc(x1)
        # DOA classification
        x2 = self.mv2_base_doa_stage2(x) # torch.Size([Batch, 48, 8, 16])
        x2 = self.mv2_base_doa_stage3(x2) # torch.Size([Batch, 96, 4, 8])
        x2 = self.avg_pool_doa(x2).squeeze(-1).squeeze(-1) # torch.Size([Batch, 96])
        out2 = self.doa_fc(x2)
        return out1, out2


"""
                               Input (Batch, 8, 128, 126)
                                             |
                                          bw2col
                                             |
                                   (Batch, 3, 128, 126)
                                             |
                                           conv1
                                             |
                                    (Batch, 24, 64, 63)
                                             |
                                          maxpool
                                             |
                                    (Batch, 24, 32, 32)
                                   /                   \
                        Frequency Branch              DOA Branch
                                 |                        |
                              stage2                    stage2
                                 |                        |
                              stage3                    stage3
                                 |                        |
                          avg_pool_fre              avg_pool_doa
                                 |                        |
                            (Batch, 96)              (Batch, 96)
                                 |                        |
                             freq_fc                    doa_fc
                                 |                        |
                  Output1 (Frequency Classes)   Output2 (DOA Classes)
"""