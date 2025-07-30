
## Installation

Download our source code:
```bash
git clone https://github.com/robot1lyj/gello_piper_lerobot.git
cd lerobot
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

When using `miniconda`, install `ffmpeg` in your environment:
```bash
conda install ffmpeg -c conda-forge
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>  - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>  ```bash
>  conda install ffmpeg=7.1.1 -c conda-forge
>  ```
>  - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

Install 🤗 LeRobot:
```bash
pip install -e .
```

> **NOTE:** If you encounter build errors, you may need to install additional dependencies (`cmake`, `build-essential`, and `ffmpeg libs`). On Linux, run:
`sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config`. For other systems, see: [Compiling PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

本仓库适配了piper机械臂基于gello的遥操作演示

- 单臂遥操作--已完成
- 完善标定程序--目前基于手动标定--待完成
- 双臂遥操作--已完成
- 完善代码架构--待完成
- 完善单臂的配置--已完成

### piper适配

#### 安装：

```shell
pip install python-can
pip install piper_sdk
sudo apt update && sudo apt install can-utils ethtool
```
下载piper-sdk仓库:https://github.com/agilexrobotics/piper_sdk

#### 机械臂使能：

##### 双臂：

先将左机械臂的can线接入电脑

然后执行：

```bash
 cd piper_sdk
 bash find_all_can_port.sh 
```

终端会出现左机械臂的端口号

```bash
ethtool 和 can-utils 均已安装。
接口 can0 插入在 USB 端口 1-3:1.0
```

接着将右机械臂的can线接入电脑

再次执行：

```bash
bash find_all_can_port.sh 
```

终端会出现左机械臂的端口号。

将这左右两个端口号复制到 can_config.sh 文件的 111 和 112 行，如下所示：

```bash
# 预定义的 USB 端口、目标接口名称及其比特率（在多个 CAN 模块时使用）
if [ "$EXPECTED_CAN_COUNT" -ne 1 ]; then
    declare -A USB_PORTS 
    USB_PORTS["1-8.1:1.0"]="left_piper:1000000"  #左机械臂
    USB_PORTS["1-8.2:1.0"]="right_piper:1000000" #右机械臂
fi
```

保存完毕后，激活左右机械臂使能脚本：

```bash
bash can_config.sh 
```

##### 单臂：

```bash
bash can_activate.sh can0 1000000
```
**注意：**这里的机械臂名称对应要到lerobot/common/robot_devices/robots/config.py 在自己新建的PiperRobotConfig中修改

### 单臂遥操作：

1.增加机械臂的驱动程序--这里用的方法相当于新增一个叫做piper的舵机--同时也可以直接调用piper的sdk
lerobot/common/robot_devices/motors/
创建piper.py 创建 class PiperMotorsBus 类实现：

```python
class MotorsBus(Protocol):
	def motor_names(self): ...
    def set_calibration(self): ...
	def apply_calibration(self): ...
	def revert_calibration(self): ...
	def read(self): ...
	def write(self): ...
```
值得注意的是，这里封装主要是对piper的sdk做二次封装。
在config.py中新增 class PiperMotorsBusConfig
utils.py中修改 def make_motors_buses_from_configs支持从PiperMotorsBusConfig中创建 PiperMotorsBus

2. 实现lerobot/common/robot_devices/robots/
这里要基于lerobot/common/robot_devices/robots/manipulator.py进行修改，主要是实现
```python
class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...
```

在config.py中新增 class PiperRobotConfig，定义camera和motor类型并配置相关参数。

在utils.py中修改 def make_robot_from_config，支持从 PiperRobotConfig 中创建PiperRobot.。

3. 标定问题
这里先根据原本的教程设置好gello的ID跟波特率
然后值得注意的是，我这里是直接写好标定的程序的，因为机械臂倒装，所以很难摆到教程的位置。
个人的一个标定的技巧：
先输出leader臂的关节数据：leader_jonits
再用sdk获得follower臂的关节数据：follower_joints
先保证两个数据的位数一样。
然后套用公式

$$
（两者差值）/360*4096
$$

填到.cache/calibration/piper/main_leader.json这里。**注意**，可以将.cache/calibration/piper/main_follower.json的offset都填为0，基于这个来调整leader。

注意的是：第一次调整完offset，可能会发生反转，那优先标定转向，再标定一次offset。

### 双臂遥操作

修改的地方也是跟单臂遥操作一样的

但是控制的频率会下降，可以增加一下机械臂的运动控制速度

### 注意事项：

1. piper的控制单位是0.001度，一般舵机控制都是两位数或者三位数，并且可以用舵机读出来的数值直接控制piper，不用从弧度转角度。
2. 倒装下的piper机械臂控制会出现很严重的抖动问题，可以用mit控制模式，可以减少很多的抖动
3. 收集数据的时候，要确保自己的datasets版本跟训练时候的版本要一致，建议3.6.0版本的




## Walkthrough

```
.
├── examples             # contains demonstration examples, start here to learn about LeRobot
|   └── advanced         # contains even more examples for those who have mastered the basics
├── lerobot
|   ├── configs          # contains config classes with all options that you can override in the command line
|   ├── common           # contains classes and utilities
|   |   ├── datasets       # various datasets of human demonstrations: aloha, pusht, xarm
|   |   ├── envs           # various sim environments: aloha, pusht, xarm
|   |   ├── policies       # various policies: act, diffusion, tdmpc
|   |   ├── robot_devices  # various real devices: dynamixel motors, opencv cameras, koch robots
|   |   └── utils          # various utilities
|   └── scripts          # contains functions to execute via command line
|       ├── eval.py                 # load policy and evaluate it on an environment
|       ├── train.py                # train a policy via imitation learning and/or reinforcement learning
|       ├── control_robot.py        # teleoperate a real robot, record data, run a policy
|       ├── push_dataset_to_hub.py  # convert your dataset into LeRobot dataset format and upload it to the Hugging Face hub
|       └── visualize_dataset.py    # load a dataset and render its demonstrations
├── outputs               # contains results of scripts execution: logs, videos, model checkpoints
└── tests                 # contains pytest utilities for continuous integration
```

### Visualize datasets

Check out [example 1](./examples/1_load_lerobot_dataset.py) that illustrates how to use our dataset class which automatically downloads data from the Hugging Face hub.

You can also locally visualize episodes from a dataset on the hub by executing our script from the command line:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

or from a dataset in a local folder with the `root` option and the `--local-files-only` (in the following case the dataset will be searched for in `./my_local_data_dir/lerobot/pusht`)
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0
```


It will open `rerun.io` and display the camera streams, robot states and actions, like this:

https://github-production-user-asset-6210df.s3.amazonaws.com/4681518/328035972-fd46b787-b532-47e2-bb6f-fd536a55a7ed.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240505%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240505T172924Z&X-Amz-Expires=300&X-Amz-Signature=d680b26c532eeaf80740f08af3320d22ad0b8a4e4da1bcc4f33142c15b509eda&X-Amz-SignedHeaders=host&actor_id=24889239&key_id=0&repo_id=748713144


Our script can also visualize datasets stored on a distant server. See `python lerobot/scripts/visualize_dataset.py --help` for more instructions.

### The `LeRobotDataset` format

A dataset in `LeRobotDataset` format is very simple to use. It can be loaded from a repository on the Hugging Face hub or a local folder simply with e.g. `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` and can be indexed into like any Hugging Face and PyTorch dataset. For instance `dataset[0]` will retrieve a single temporal frame from the dataset containing observation(s) and an action as PyTorch tensors ready to be fed to a model.

A specificity of `LeRobotDataset` is that, rather than retrieving a single frame by its index, we can retrieve several frames based on their temporal relationship with the indexed frame, by setting `delta_timestamps` to a list of relative times with respect to the indexed frame. For example, with `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}`  one can retrieve, for a given index, 4 frames: 3 "previous" frames 1 second, 0.5 seconds, and 0.2 seconds before the indexed frame, and the indexed frame itself (corresponding to the 0 entry). See example [1_load_lerobot_dataset.py](examples/1_load_lerobot_dataset.py) for more details on `delta_timestamps`.

Under the hood, the `LeRobotDataset` format makes use of several ways to serialize data which can be useful to understand if you plan to work more closely with this format. We tried to make a flexible yet simple dataset format that would cover most type of features and specificities present in reinforcement learning and robotics, in simulation and in real-world, with a focus on cameras and robot states but easily extended to other types of sensory inputs as long as they can be represented by a tensor.

Here are the important details and internal structure organization of a typical `LeRobotDataset` instantiated with `dataset = LeRobotDataset("lerobot/aloha_static_coffee")`. The exact features will change from dataset to dataset but not the main aspects:

```
dataset attributes:
  ├ hf_dataset: a Hugging Face dataset (backed by Arrow/parquet). Typical features example:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': path to a mp4 video, 'timestamp' (float32): timestamp in the video}
  │  ├ observation.state (list of float32): position of an arm joints (for instance)
  │  ... (more observations)
  │  ├ action (list of float32): goal position of an arm joints (for instance)
  │  ├ episode_index (int64): index of the episode for this sample
  │  ├ frame_index (int64): index of the frame for this sample in the episode ; starts at 0 for each episode
  │  ├ timestamp (float32): timestamp in the episode
  │  ├ next.done (bool): indicates the end of an episode ; True for the last frame in each episode
  │  └ index (int64): general index in the whole dataset
  ├ episode_data_index: contains 2 tensors with the start and end indices of each episode
  │  ├ from (1D int64 tensor): first frame index for each episode — shape (num episodes,) starts with 0
  │  └ to: (1D int64 tensor): last frame index for each episode — shape (num episodes,)
  ├ stats: a dictionary of statistics (max, mean, min, std) for each feature in the dataset, for instance
  │  ├ observation.images.cam_high: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  ...
  ├ info: a dictionary of metadata on the dataset
  │  ├ codebase_version (str): this is to keep track of the codebase version the dataset was created with
  │  ├ fps (float): frame per second the dataset is recorded/synchronized to
  │  ├ video (bool): indicates if frames are encoded in mp4 video files to save space or stored as png files
  │  └ encoding (dict): if video, this documents the main options that were used with ffmpeg to encode the videos
  ├ videos_dir (Path): where the mp4 videos or png images are stored/accessed
  └ camera_keys (list of string): the keys to access camera features in the item returned by the dataset (e.g. `["observation.images.cam_high", ...]`)
```

A `LeRobotDataset` is serialised using several widespread file formats for each of its parts, namely:
- hf_dataset stored using Hugging Face datasets library serialization to parquet
- videos are stored in mp4 format to save space
- metadata are stored in plain json/jsonl files

Dataset can be uploaded/downloaded from the HuggingFace hub seamlessly. To work on a local dataset, you can specify its location with the `root` argument if it's not in the default `~/.cache/huggingface/lerobot` location.

### Evaluate a pretrained policy

Check out [example 2](./examples/2_evaluate_pretrained_policy.py) that illustrates how to download a pretrained policy from Hugging Face hub, and run an evaluation on its corresponding environment.

We also provide a more capable script to parallelize the evaluation over multiple environments during the same rollout. Here is an example with a pretrained model hosted on [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht):
```bash
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note: After training your own policy, you can re-evaluate the checkpoints with:

```bash
python lerobot/scripts/eval.py --policy.path={OUTPUT_DIR}/checkpoints/last/pretrained_model
```

See `python lerobot/scripts/eval.py --help` for more instructions.

### Train your own policy

Check out [example 3](./examples/3_train_policy.py) that illustrates how to train a model using our core library in python, and [example 4](./examples/4_train_policy_with_script.md) that shows how to use our training script from command line.

To use wandb for logging training and evaluation curves, make sure you've run `wandb login` as a one-time setup step. Then, when running the training command above, enable WandB in the configuration by adding `--wandb.enable=true`.

A link to the wandb logs for the run will also show up in yellow in your terminal. Here is an example of what they look like in your browser. Please also check [here](./examples/4_train_policy_with_script.md#typical-logs-and-metrics) for the explanation of some commonly used metrics in logs.

![](media/wandb.png)

Note: For efficiency, during training every checkpoint is evaluated on a low number of episodes. You may use `--eval.n_episodes=500` to evaluate on more episodes than the default. Or, after training, you may want to re-evaluate your best checkpoints on more episodes or change the evaluation settings. See `python lerobot/scripts/eval.py --help` for more instructions.

#### Reproduce state-of-the-art (SOTA)

We provide some pretrained policies on our [hub page](https://huggingface.co/lerobot) that can achieve state-of-the-art performances.
You can reproduce their training by loading the config from their run. Simply running:
```bash
python lerobot/scripts/train.py --config_path=lerobot/diffusion_pusht
```
reproduces SOTA results for Diffusion Policy on the PushT task.

## Contribute

If you would like to contribute to 🤗 LeRobot, please check out our [contribution guide](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md).

<!-- ### Add a new dataset

To add a dataset to the hub, you need to login using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Then point to your raw dataset folder (e.g. `data/aloha_static_pingpong_test_raw`), and push your dataset to the hub with:
```bash
python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir data/aloha_static_pingpong_test_raw \
--out-dir data \
--repo-id lerobot/aloha_static_pingpong_test \
--raw-format aloha_hdf5
```

See `python lerobot/scripts/push_dataset_to_hub.py --help` for more instructions.

If your dataset format is not supported, implement your own in `lerobot/common/datasets/push_dataset_to_hub/${raw_format}_format.py` by copying examples like [pusht_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py), [umi_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/umi_zarr_format.py), [aloha_hdf5](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py), or [xarm_pkl](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/xarm_pkl_format.py). -->


### Add a pretrained policy

Once you have trained a policy you may upload it to the Hugging Face hub using a hub id that looks like `${hf_user}/${repo_name}` (e.g. [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)).

You first need to find the checkpoint folder located inside your experiment directory (e.g. `outputs/train/2024-05-05/20-21-12_aloha_act_default/checkpoints/002500`). Within that there is a `pretrained_model` directory which should contain:
- `config.json`: A serialized version of the policy configuration (following the policy's dataclass config).
- `model.safetensors`: A set of `torch.nn.Module` parameters, saved in [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) format.
- `train_config.json`: A consolidated configuration containing all parameters used for training. The policy configuration should match `config.json` exactly. This is useful for anyone who wants to evaluate your policy or for reproducibility.

To upload these to the hub, run the following:
```bash
huggingface-cli upload ${hf_user}/${repo_name} path/to/pretrained_model
```

See [eval.py](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py) for an example of how other people may use your policy.


### Improve your code with profiling

An example of a code snippet to profile the evaluation of a policy:
```python
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(prof):
    prof.export_chrome_trace(f"tmp/trace_schedule_{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=3,
    ),
    on_trace_ready=trace_handler
) as prof:
    with record_function("eval_policy"):
        for i in range(num_episodes):
            prof.step()
            # insert code to profile, potentially whole body of eval_policy function
```

## Citation

If you want, you can cite this work with:
```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascale, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

Additionally, if you are using any of the particular policy architecture, pretrained models, or datasets, it is recommended to cite the original authors of the work as they appear below:

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu)
```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
- [ACT or ALOHA](https://tonyzhaozh.github.io/aloha)
```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

- [TDMPC](https://www.nicklashansen.com/td-mpc/)

```bibtex
@inproceedings{Hansen2022tdmpc,
	title={Temporal Difference Learning for Model Predictive Control},
	author={Nicklas Hansen and Xiaolong Wang and Hao Su},
	booktitle={ICML},
	year={2022}
}
```

- [VQ-BeT](https://sjlee.cc/vq-bet/)
```bibtex
@article{lee2024behavior,
  title={Behavior generation with latent actions},
  author={Lee, Seungjae and Wang, Yibin and Etukuru, Haritheja and Kim, H Jin and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2403.03181},
  year={2024}
}
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=huggingface/lerobot&type=Timeline)](https://star-history.com/#huggingface/lerobot&Timeline)
