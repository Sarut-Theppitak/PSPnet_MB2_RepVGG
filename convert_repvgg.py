from encoder.repvgg import whole_model_convert
import torch
from model import PSPNet

# CLASSES = ['car']
# train_pspnet = torch.load('./models/repvggA2_g16_train.pth')
# deploy_pspnet = PSPNet(
#             preupsample=False,
#             upsampling=8,
#             encoder_name="repvgg",
#             encoder_weights=False,
#             encoder_depth=5,
#             psp_out_channels=512,             
#             psp_use_batchnorm=True,
#             psp_dropout=0.2,
#             in_channels=3,
#             classes=len(CLASSES),
#             activation='sigmoid',   
#             dilated=False,
#             deploy=True
#         )

# whole_model_convert(train_pspnet, deploy_pspnet)
# torch.save(deploy_pspnet, './models/repvggA2_g16_deploy.pth')




################################# test eqivalent ##################################################


train_pspnet = torch.load('./models/repvggA2_g16_train.pth').cuda()
deploy_pspnet = torch.load('./models/repvggA2_g16_deploy.pth').cuda()
train_pspnet.eval()
deploy_pspnet.eval()

x = torch.randn(1, 3, 224, 224).cuda()

train_y = train_pspnet(x)
deploy_y = deploy_pspnet(x)
print(((train_y - deploy_y) ** 2).sum()) 