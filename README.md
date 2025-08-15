# Next-Frame Prediction Using Convolutional LSTM


<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/next-frame-prediction/blob/master/Next_Frame_Prediction_Using_Convolutional_LSTM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>


In the next-frame prediction problem, we strive to generate the subsequent frame of a given video. Inherently, video has two kinds of information to take into account, i.e., image (spatial) and temporal. Using the Convolutional LSTM model, we can manage to feature-extract and process both pieces of information with their inductive biases. In Convolutional LSTM, instead of utilizing fully connected layers within the LSTM cell, convolution operations are adopted. To evaluate the model, the moving MNIST dataset is used. To evalute the model, the Moving MNIST dataset is used.


## Experiment

Have a dive into this [link](https://colab.research.google.com/github/reshalfahsi/next-frame-prediction/blob/master/Next_Frame_Prediction_Using_Convolutional_LSTM.ipynb) and immerse yourself in the next-frame prediction implementation.


## Result

## Quantitative Result

Inspect this table to catch sight of the model's feat.

Test Metric  | Score
------------ | -------------
Loss         | 0.006
MAE          | 0.021
PSNR         | 22.120
SSIM         | 0.881


## Evaluation Metric Curve

<p align="center"> <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> The loss curve on the training and validation sets of the Convolutional LSTM model. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/mae_curve.png" alt="mae_curve" > <br /> The MAE curve on the training and validation sets of the Convolutional LSTM model. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/psnr_curve.png" alt="psnr_curve" > <br /> The PSNR curve on the training and validation sets of the Convolutional LSTM model. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/ssim_curve.png" alt="ssim_curve" > <br /> The SSIM curve on the training and validation sets of the Convolutional LSTM model. </p>


## Qualitative Result

This GIF displays the qualitative result of the frame-by-frame prediction of the Convolutional LSTM model.

<p align="center"> <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/result.gif" alt="qualitative" > <br /> The Convolutional LSTM model predicts the ensuing frame-by-frame from <i>t</i> = 1 to <i>t</i> = 19. </p>


## Credit

- [Next-Frame Video Prediction with Convolutional LSTMs](https://keras.io/examples/vision/conv_lstm/)
- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf)
- [Statistical Language Models Based on Neural Networks](https://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
- [Unsupervised Learning of Video Representations using LSTMs](http://www.cs.toronto.edu/~nitish/unsup_video.pdf)
- [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech Recognition](https://arxiv.org/pdf/1402.1128.pdf)
- [Forked Torchvision by Henry Xia](https://github.com/ehnryx/vision/tree/be6f398c0612c245b0019a286a99f80aca81de7d/torchvision/transforms)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
