# Colonoscopy-Depth-Estimation

As part of [SimCol-to-3D 2022](https://www.synapse.org/#!Synapse:syn28548633/wiki/617134), the Depth Estimation task was carried out to predict accurate depth from simulated colonoscopy images. This approach proposes to use a fully convolutional neural network SUMNet which consists of: 
1. an encoder-decoder type architecture with the transfer of pooling indices from the encoder to the decoder upsampling layer at the matched depth
2. an encoder network with VGG11 architecture initialised with ImageNet pre-trained weights
3. activation concatenation from the encoder to the decoder.

The training code is available in SUMNet-depth-training.py and requires the training, validation and test splits (refer code). Once a model is trained, SUMNet-depth-test.py can be used with any sample input images to give the output images with predicted depth. 
