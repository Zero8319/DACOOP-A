# DACOOP-A

![DACOOP-A](https://github.com/Zero8319/DACOOP-A/blob/main/Images/DACOOP_A.png)

This repository includes codes, figures, and videos for the algorithm Decentralized Adaptive COOperative Pursuit via Attention (DACOOP-A) in the paper titled "DACOOP-A: Decentralized Adaptive Cooperative Pursuit via Attention".

To run the code, simply execute: python3 attention3.py.

To visualize the learned policies, execute: python3 validation.py.

./Codes/Results/4000.pt is the weights of Q network that DACOOP-A learns. For details, please check the output of attention3.py.

./Codes/Results/4000_time.txt is the validation results over 1000 episodes, where 1000 means failure and any number less than 1000 means success. For details, please check the output of validation.py.

# Dependencies
- python 3.7
- numpy 1.19
- torch 1.8
- matplotlib 3.3


