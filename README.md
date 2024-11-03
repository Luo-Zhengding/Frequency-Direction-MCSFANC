# Frequency-Direction-MCSFANC

This repository contains the code for the paper "**Frequency-Direction Aware Multichannel Selective Fixed-Filter Active Noise Control based on Multi-Task Learning**," submitted to the **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)** journal. The code will be provided after the paper review process is completed.

This is a collaborative research work between the Digital Signal Processing Lab at Nanyang Technological University and Zhejiang University.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e2b5661-e3b8-4cfe-b25e-b784be1dffe4" width="500"><br>
  The framework of the proposed FD-MCSFANC method
</p>

<br> <!-- 添加空行 -->

<p align="center">
  <img src="https://github.com/Luo-Zhengding/GFANC-RL/assets/95018034/c44ad8c5-dafb-4811-b169-fb3ebdafd9ec" width="400"> 
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <!-- 使用空格来创建间隔 -->
  <img src="https://github.com/Luo-Zhengding/GFANC-RL/assets/95018034/04d1ab12-beb7-4123-b680-4cf8b91d3173" width="400">
  <br>
  Parameter update for the critic and actor in the RL algorithm.
</p>


## Highlights
1. GFANC-RL employs RL techniques to address challenges associated with GFANC innovatively.
2. This paper formulates the GFANC problem as a Markov Decision Process (MDP) from a decision-making perspective, laying a theoretical foundation for using RL algorithms.
3. In the GFANC-RL method, an RL algorithm based on Soft Actor-Critic (SAC) is developed to train the CNN using unlabelled noise data and improve the exploration ability of the CNN model.
4. Experimental results show that the GFANC-RL method effectively attenuates real-recorded noises and exhibits good robustness and transferability in different acoustic paths.
