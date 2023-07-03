"""
SimCol-to-3D challenge - MICCAI 2022
Challenge link: https://www.synapse.org/#!Synapse:syn28548633/wiki/617126
Task 1: Depth prediction in simulated colonoscopy
Task 2: Camera pose estimation in simulated colonoscopy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import models
from torchvision import transforms 
import numpy as np
from PIL import Image
import os
import glob

to_tensor = transforms.ToTensor()

class conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SUMNet(nn.Module):
    def __init__(self):
        super(SUMNet, self).__init__()
        
        self.encoder   = models.vgg11(pretrained = True).features
        self.conv1     = nn.Sequential(
                            self.encoder[0],
                            nn.BatchNorm2d(64)
                        )
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv2     = nn.Sequential(
                            self.encoder[3],
                            nn.BatchNorm2d(128)
                        )
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = nn.Sequential(
                            self.encoder[6],
                            nn.BatchNorm2d(256)
                        )
        self.conv3b    = nn.Sequential(
                            self.encoder[8],
                            nn.BatchNorm2d(256)
                        )
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = nn.Sequential(
                            self.encoder[11],
                            nn.BatchNorm2d(512)
                        )
        self.conv4b    = nn.Sequential(
                            self.encoder[13],
                            nn.BatchNorm2d(512)
                        )
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = nn.Sequential(
                            self.encoder[16],
                            nn.BatchNorm2d(512)
                        )
        self.conv5b    = nn.Sequential(
                            self.encoder[18],
                            nn.BatchNorm2d(512)
                        )
        self.pool5     = nn.MaxPool2d(2, 2, return_indices = True)
        
        self.unpool5   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = conv_bn(1024, 512)
        self.donv5a    = conv_bn(512, 512)
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = conv_bn(1024, 512)
        self.donv4a    = conv_bn(512, 256)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = conv_bn(512, 256)
        self.donv3a    = conv_bn(256,128)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv2     = conv_bn(256, 64)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv1     = conv_bn(128, 32)
        self.output    = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        
        conv1          = F.relu(self.conv1(x), inplace = True)
        pool1, idxs1   = self.pool1(conv1)
        conv2          = F.relu(self.conv2(pool1), inplace = True)
        pool2, idxs2   = self.pool2(conv2)
        conv3a         = F.relu(self.conv3a(pool2), inplace = True)
        conv3b         = F.relu(self.conv3b(conv3a), inplace = True)
        pool3, idxs3   = self.pool3(conv3b)
        conv4a         = F.relu(self.conv4a(pool3), inplace = True)
        conv4b         = F.relu(self.conv4b(conv4a), inplace = True)
        pool4, idxs4   = self.pool4(conv4b)
        conv5a         = F.relu(self.conv5a(pool4), inplace = True)
        conv5b         = F.relu(self.conv5b(conv5a), inplace = True)
        pool5, idxs5   = self.pool5(conv5b)
        
        unpool5        = torch.cat([self.unpool5(pool5, idxs5), conv5b], 1)
        donv5b         = F.relu(self.donv5b(unpool5), inplace = True)
        donv5a         = F.relu(self.donv5a(donv5b), inplace = True)
        unpool4        = torch.cat([self.unpool4(donv5a, idxs4), conv4b], 1)
        donv4b         = F.relu(self.donv4b(unpool4), inplace = True)
        donv4a         = F.relu(self.donv4a(donv4b), inplace = True)
        unpool3        = torch.cat([self.unpool3(donv4a, idxs3), conv3b], 1)
        donv3b         = F.relu(self.donv3b(unpool3), inplace = True)
        donv3a         = F.relu(self.donv3a(donv3b))
        unpool2        = torch.cat([self.unpool2(donv3a, idxs2), conv2], 1)
        donv2          = F.relu(self.donv2(unpool2), inplace = True)
        unpool1        = torch.cat([self.unpool1(donv2, idxs1), conv1], 1)
        donv1          = F.relu(self.donv1(unpool1), inplace = True)
        output         = self.output(donv1)
        
        return torch.sigmoid(output)

mean_f = torch.tensor([0.7050, 0.4508, 0.2961])
std_f = torch.tensor([0.2382, 0.2037, 0.1276])
transform_frame = transforms.Compose([
                      transforms.Resize(448),
                      transforms.ToTensor(),
                      transforms.Normalize(mean_f,std_f)       
                ])
resize_op = transforms.Resize(475)

model = SUMNet()
device = torch.device("cpu") # or "cuda"
model = model.to(device)

# Give the path to the final trained model 
checkpoint_path = 'checkpoint_50.pt'
checkpoint = torch.load(checkpoint_path,map_location = torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def read_inp_img(im_path):
    frame = Image.open(im_path).convert('RGB')
    frame = transform_frame(frame)
    return frame

def predict_depth(im_path, output_folder):
    """
    param im_path: Input path for a single image
    param output_folder: Path to folder where output will be saved
    predict the depth for an image and save in the correct formatting as .npy file
    """

    frame = read_inp_img(im_path)
    frame = frame.to(device)
    gt_depth = to_tensor(Image.open(im_path.replace('FrameBuffer','Depth'))).squeeze()/255/256 # please use this to load ground truth depth during training
    prediction = model(frame.unsqueeze(0))
    predicted_depth = resize_op(prediction).squeeze().detach().cpu().numpy()
    # The output depth should be in the range [0,1] similar to the input format. Note that a depth of 1 of the output
    # depth should correspond to 20cm in the world. The organizers will clip all values <0 and >1 to a valid range [0,1]
    # and multiply the output by 20 to obtain depth in centimeters.

    ### Output and save your prediction in the correct format ###
    out_file = im_path.split('/')[-1]
    out_file = out_file.replace('.png','')
    assert predicted_depth.shape == (475,475), \
        "Wrong size of predicted depth, expected [475,475], got {}".format(list(predicted_depth.shape))

    np.save(output_folder + out_file, np.float16(predicted_depth))  # save a np.float16() to reduce file size
    print(output_folder + out_file + '.npy saved')
    # Note: Saving as np.float16() will lead to minor loss in precision


    ### Double check that the organizers' evaluation pipeline will correctly reload your depths (uncomment below) ###
    reloaded_prediction = torch.from_numpy(np.load(output_folder + out_file + '.npy'))
    print('Half precision error: {} cm'.format(np.max((reloaded_prediction.numpy() - predicted_depth)) * 20))
    error = torch.mean(torch.abs(reloaded_prediction - gt_depth)) * 20  #Note that this is for illustration only. Validation pipeline will be published separately.
    print('Mean error: {} cm'.format(error.numpy()))

# to run without docker
INPUT_PATH = '../images/input' #save sample test images in this format
OUTPUT_PATH = '../images/output' #directory to write the output images to

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(OUTPUT_PATH +' created')
    os.makedirs(OUTPUT_PATH+'/depth/')
else:
    print(OUTPUT_PATH +' exists')

if not os.path.exists(OUTPUT_PATH+'/depth/'):
    os.makedirs(OUTPUT_PATH+'/depth/')

if not os.path.exists(INPUT_PATH):
    print('No input folder found')
else:
    print(INPUT_PATH +' exists')
    glob.__file__
    input_file_list = np.sort(glob.glob(INPUT_PATH + "/FrameBuffer*.png"))

for i in range(len(input_file_list)):
    file_name1 = input_file_list[i].split("/")[-1]
    im1_path = INPUT_PATH + '/' + file_name1
    predict_depth(im1_path, OUTPUT_PATH+'/depth/')


