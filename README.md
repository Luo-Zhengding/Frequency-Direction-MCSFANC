# Frequency-Direction-MCSFANC

This repository contains the code for the paper "**Frequency-Direction Aware Multichannel Selective Fixed-Filter Active Noise Control based on Multi-Task Learning**," published in **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)** journal. The paper link is [Paper Link](https://ieeexplore.ieee.org/document/11082568).

This is a collaborative research work between the Digital Signal Processing Lab at Nanyang Technological University, Northwestern Polytechnical University, and Zhejiang University.

## Research Background
The Selective Fixed-filter Active Noise Control (SFANC) method has gained popularity due to its high computational efficiency and fast response. However, it solely accounts for noise frequency information, while neglecting its spatial information, which undoubtedly undermines the noise reduction performance, especially for direction-varied noises. To address this limitation, we proposed the **Frequency-Direction Aware Multichannel SFANC (FD-MCSFANC)** method, which incorporates **both frequency and direction information** of primary noises for accurate filter selection in the multichannel ANC (MCANC) system.

## FD-MCSFANC Framework
<p align="center">
  <img src="https://github.com/user-attachments/assets/6e2b5661-e3b8-4cfe-b25e-b784be1dffe4" width="600"><br>
  The framework of the proposed FD-MCSFANC method
</p>

<br>

## Network Training and Inference
<div align="center">
  <div style="display: inline-block;">
    <img src="https://github.com/user-attachments/assets/3f68a5bd-1b91-4e52-b23d-baad618df229" width="500">
    <br>
    <span style="font-weight: bold;">(a) End-to-end training of the CNN based on multi-task learning.</span>
  </div>
  <div style="display: inline-block;">
    <img src="https://github.com/user-attachments/assets/a624c860-9048-4c30-a1f2-f897ec3927d0" width="400">
    <br>
    <span style="font-weight: bold;">(b) Filter selection process using the trained CNN.</span>
  </div>
</div>

&nbsp;
- In the FD-MCSFANC method, a lightweight CNN is designed to classify noises based on both frequency components and Direction-of-Arrival (DOA), with the combined classification results determining selected control filters.
- Furthermore, a joint loss function based on multi-task learning is utilized to implement end-to-end training of the CNN.
- Numerical simulations show that the FD-MCSFANC method effectively attenuates noises with different frequencies and incident angles. Moreover, it responds much faster than traditional adaptive algorithms and achieves better global noise control performance than the Multichannel SFANC (MCSFANC) method.

## Related Publications
1. **Real-time implementation and explainable AI analysis of delayless CNN-based selective fixed-filter active noise control**  
   *Journal*: Mechanical Systems and Signal Processing, 2024, 214: 111364.  
   *Paper Link*: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327024002620) &nbsp; *Code Link*: [GitHub](https://github.com/Luo-Zhengding/SFANC-Window)

2. **A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control based on Deep Learning**  
   *Journal*: IEEE Signal Processing Letters, 2022, 29: 1102-1106.
   *Paper Link*: [IEEE](https://ieeexplore.ieee.org/document/9761749) &nbsp; *Code Link*: [GitHub](https://github.com/Luo-Zhengding/SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning)

3. **Performance Evaluation of Selective Fixed-filter Active Noise Control based on Different Convolutional Neural Networks**  
   *Conference*: The 51st International Congress and Exposition on Noise Control Engineering (Inter-Noise 2022)
   *Paper Link*: [arXiv](https://arxiv.org/pdf/2208.08440)

**If you are interested in our works, please consider citing our papers. Thanks! Have a great day!**
