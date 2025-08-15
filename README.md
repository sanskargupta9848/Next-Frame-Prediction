# Next-Frame Prediction with Convolutional LSTM

<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/next-frame-prediction/blob/master/Next_Frame_Prediction_Using_Convolutional_LSTM.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
    </a>
</div>

## Overview

This project tackles the **next-frame prediction** challenge — generating the following frame in a video sequence.  
Videos contain both **spatial** (image) and **temporal** information, and Convolutional LSTM (ConvLSTM) models can process both simultaneously.  
Unlike traditional LSTMs that use fully connected layers, ConvLSTMs employ **convolutional operations** within their recurrent units, making them ideal for video-based tasks.

The **Moving MNIST** dataset is used for model training and evaluation.

## Try It Out

Explore the live implementation here:  
[**Google Colab Notebook**](https://colab.research.google.com/github/reshalfahsi/next-frame-prediction/blob/master/Next_Frame_Prediction_Using_Convolutional_LSTM.ipynb)

## Results

### Quantitative Metrics

| Metric       | Score   |
|--------------|---------|
| Loss         | 0.006   |
| MAE          | 0.021   |
| PSNR         | 22.120  |
| SSIM         | 0.881   |

### Training Curves

<p align="center">
  <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/loss_curve.png" alt="Loss Curve"><br>
  <em>Loss on training and validation sets</em>
</p>

<p align="center">
  <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/mae_curve.png" alt="MAE Curve"><br>
  <em>Mean Absolute Error (MAE) on training and validation sets</em>
</p>

<p align="center">
  <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/psnr_curve.png" alt="PSNR Curve"><br>
  <em>Peak Signal-to-Noise Ratio (PSNR) on training and validation sets</em>
</p>

<p align="center">
  <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/ssim_curve.png" alt="SSIM Curve"><br>
  <em>Structural Similarity Index (SSIM) on training and validation sets</em>
</p>

### Qualitative Results

<p align="center">
  <img src="https://github.com/reshalfahsi/next-frame-prediction/blob/master/assets/result.gif" alt="Qualitative Result"><br>
  <em>Frame-by-frame predictions from <i>t</i> = 1 to <i>t</i> = 19</em>
</p>

## References

- [Next-Frame Video Prediction with Convolutional LSTMs](https://keras.io/examples/vision/conv_lstm/)
- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)
- [On the Difficulty of Training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf)
- [Statistical Language Models Based on Neural Networks](https://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
- [Unsupervised Learning of Video Representations using LSTMs](http://www.cs.toronto.edu/~nitish/unsup_video/)
- [Moving MNIST Dataset](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [LSTM-based RNN Architectures for Large Vocabulary Speech Recognition](https://arxiv.org/pdf/1402.1128.pdf)
- [Forked Torchvision by Henry Xia](https://github.com/ehnryx/vision/tree/be6f398c0612c245b0019a286a99f80aca81de7d/torchvision/transforms)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
