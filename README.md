# Knowledge Transfer in Multi-Task Deep Reinforcement Learning for Continuous Control

This repository is the official implementation of [KTM-DRL](https://arxiv.org/abs/2010.07494). 

## Dependencies

- Python 3.7.7 (Optional, also works with older versions)
- [PyTorch 1.5.1](https://github.com/pytorch/pytorch)
- [MuJoCo 200](http://www.mujoco.org/index.html)
- [mujoco-py 2.0.2.5](https://github.com/openai/mujoco-py)
- [OpenAI Gym 0.17.2](https://github.com/openai/gym)
- [gym-extensions](https://github.com/Breakend/gym-extensions) (Optional, only for HalfCheetah task group)

## Evaluation

To use the pre-trained models of task-specific teachers and multi-task agent on HalfCheetah task group, 
download from the [Dropbox Link](https://www.dropbox.com/sh/gs7n4g9f27izubn/AADhX5sts3UK-Fsm0xPvWHoza?dl=0) and put the `half` folder into `./model/` or `/your_own_path/`. 


To evaluate the multi-task agent and its corresponding task-specific teachers, run this command:

```eval
python3 eval-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-eval.json --dir ./model/half --name EVA
```


## Training

To train the model(s) in the paper, run this command:

```train
python3 train-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-train.json --dir ./model/half --name TRN
```


### Bibtex

```
@article{xu2020knowledge,
  title={Knowledge Transfer in Multi-Task Deep Reinforcement Learning for Continuous Control},
  author={Xu, Zhiyuan and Wu, Kun and Che, Zhengping and Tang, Jian and Ye, Jieping},
  journal={arXiv preprint arXiv:2010.07494},
  year={2020}
}
``` 
