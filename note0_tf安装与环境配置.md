# TensorFlow 安装与环境配置

## 安装常用命令

### pip

```python
pip install [package-name]              # 安装名为[package-name]的包
pip install [package-name]==X.X         # 安装名为[package-name]的包并指定版本X.X
pip install [package-name] --proxy=代理服务器IP:端口号         # 使用代理服务器安装
pip install [package-name] --upgrade    # 更新名为[package-name]的包
pip uninstall [package-name]            # 删除名为[package-name]的包
pip list                                # 列出当前环境下已安装的所有包
```
### conda

```
conda install [package-name]        # 安装名为[package-name]的包
conda install [package-name]=X.X    # 安装名为[package-name]的包并指定版本X.X
conda update [package-name]         # 更新名为[package-name]的包
conda remove [package-name]         # 删除名为[package-name]的包
conda list                          # 列出当前环境下已安装的所有包
conda search [package-name]         # 列出名为[package-name]的包在conda源中的所有可用版本
```

conda 中配置代理：在用户目录下的 .condarc 文件中添加以下内容：

```
proxy_servers:
    http: http://代理服务器IP:端口号
```

### conda创建环境

```
conda create --name [env-name]      # 建立名为[env-name]的Conda虚拟环境
conda activate [env-name]           # 进入名为[env-name]的Conda虚拟环境
conda deactivate                    # 退出当前的Conda虚拟环境
conda env remove --name [env-name]  # 删除名为[env-name]的Conda虚拟环境
conda env list                      # 列出所有Conda虚拟环境
```

## GPU 版本 TensorFlow 安装指南 

GPU 版本的 TensorFlow 可以利用 NVIDIA GPU 强大的计算加速能力，使 TensorFlow 的运行更为高效，尤其是可以成倍提升模型训练的速度。

在安装 GPU 版本的 TensorFlow 前，你需要具有一块不太旧的 NVIDIA 显卡，以及正确安装 NVIDIA 显卡驱动程序、CUDA Toolkit 和 cuDNN。

### GPU 硬件的准备 

TensorFlow 对 NVIDIA 显卡的支持较为完备。对于 NVIDIA 显卡，要求其 CUDA Compute Capability 须不低于 3.0，可以到 [NVIDIA 的官方网站](https://developer.nvidia.com/cuda-gpus/) 查询自己所用显卡的 CUDA Compute Capability。目前，AMD 的显卡也开始对 TensorFlow 提供支持，可访问 [这篇博客文章](https://medium.com/tensorflow/amd-rocm-gpu-support-for-tensorflow-33c78cc6a6cf) 查看详情。

### NVIDIA 驱动程序的安装 

**Windows**

Windows 环境中，如果系统具有 NVIDIA 显卡，则往往已经自动安装了 NVIDIA 显卡驱动程序。如未安装，直接访问 [NVIDIA 官方网站](https://www.nvidia.com/Download/index.aspx?lang=en-us) 下载并安装对应型号的最新公版驱动程序即可。

**Linux**

在服务器版 Linux 系统下，同样访问 [NVIDIA 官方网站](https://www.nvidia.com/Download/index.aspx?lang=en-us) 下载驱动程序（为 `.run` 文件），并使用 `sudo bash DRIVER_FILE_NAME.run` 命令安装驱动即可。在安装之前，可能需要使用 `sudo apt-get install build-essential` 安装合适的编译环境。

在具有图形界面的桌面版 Linux 系统上，NVIDIA 显卡驱动程序需要一些额外的配置，否则会出现无法登录等各种错误。如果需要在 Linux 下手动安装 NVIDIA 驱动，注意在安装前进行以下步骤（以 Ubuntu 为例）：

- 禁用系统自带的开源显卡驱动 Nouveau（在 `/etc/modprobe.d/blacklist.conf` 文件中添加一行 `blacklist nouveau` ，使用 `sudo update-initramfs -u` 更新内核，并重启）
- 禁用主板的 Secure Boot 功能
- 停用桌面环境（如 `sudo service lightdm stop`）
- 删除原有 NVIDIA 驱动程序（如 `sudo apt-get purge nvidia*`）

小技巧

对于桌面版 Ubuntu 系统，有一个很简易的 NVIDIA 驱动安装方法：在系统设置（System Setting）里面选软件与更新（Software & Updates），然后点选 Additional Drivers 里面的 “Using NVIDIA binary driver” 选项并点选右下角的 “Apply Changes” 即可，系统即会自动安装 NVIDIA 驱动，但是通过这种安装方式安装的 NVIDIA 驱动往往版本较旧。

NVIDIA 驱动程序安装完成后，可在命令行下使用 `nvidia-smi` 命令检查是否安装成功，若成功则会打印出当前系统安装的 NVIDIA 驱动信息，形式如下：

```
$ nvidia-smi
Mon Jun 10 23:19:54 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 419.35       Driver Version: 419.35       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 106... WDDM  | 00000000:01:00.0  On |                  N/A |
| 27%   51C    P8    13W / 180W |   1516MiB /  6144MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0       572    C+G   Insufficient Permissions                   N/A      |
+-----------------------------------------------------------------------------+
```

提示

命令 `nvidia-smi` 可以查看机器上现有的 GPU 及使用情况。（在 Windows 下，将 `C:\Program Files\NVIDIA Corporation\NVSMI` 加入 Path 环境变量中即可，或 Windows 10 下可使用任务管理器的 “性能” 标签查看显卡信息）

更详细的 GPU 环境配置指导可以参考 [这篇文章](https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/) 和 [这篇中文博客](https://blog.csdn.net/wf19930209/article/details/81877822) 。

### CUDA Toolkit 和 cuDNN 的安装 

在 Anaconda 环境下，推荐使用

```
conda install cudatoolkit=X.X
conda install cudnn=X.X.X
```

安装 CUDA Toolkit 和 cuDNN，其中 X.X 和 X.X.X 分别为需要安装的 CUDA Toolkit 和 cuDNN 版本号，必须严格按照 TensorFlow 官方网站所说明的版本安装。在安装前，可使用 `conda search cudatoolkit` 和 `conda search cudnn` 搜索 conda 源中可用的版本号。

当然，也可以按照 [TensorFlow 官方网站上的说明](https://www.tensorflow.org/install/gpu) 手动下载 CUDA Toolkit 和 cuDNN 并安装，不过过程会稍繁琐。

使用 conda 包管理器安装 GPU 版本的 TensorFlow 时，会自动安装对应版本的 CUDA Toolkit 和 cuDNN。conda 源的更新较慢，如果对版本不太介意，推荐直接使用 `conda install tensorflow-gpu` 进行安装。

## IDE 设置 

对于机器学习的研究者和从业者，建议使用 [PyCharm](http://www.jetbrains.com/pycharm/) 作为 Python 开发的 IDE。

在新建项目时，你需要选定项目的 Python Interpreter，也就是用怎样的 Python 环境来运行你的项目。在安装部分，你所建立的每个 Conda 虚拟环境其实都有一个自己独立的 Python Interpreter，你只需要将它们添加进来即可。选择 “Add”，并在接下来的窗口选择 “Existing Environment”，在 Interpreter 处选择 `Anaconda安装目录/envs/所需要添加的Conda环境名字/python.exe` （Linux 下无 `.exe` 后缀）并点击 “OK” 即可。如果选中了 “Make available to all projects”，则在所有项目中都可以选择该 Python Interpreter。注意，在 Windows 下 Anaconda 的默认安装目录比较特殊，一般为 `C:\Users\用户名\Anaconda3\` 或 `C:\Users\用户名\AppData\Local\Continuum\anaconda3` 。此处 `AppData` 是隐藏文件夹。

对于 TensorFlow 开发而言，PyCharm 的 Professonal 版本非常有用的一个特性是 **远程调试** （Remote Debugging）。当你编写程序的终端机性能有限，但又有一台可远程 ssh 访问的高性能计算机（一般具有高性能 GPU）时，远程调试功能可以让你在终端机编写程序的同时，在远程计算机上调试和运行程序（尤其是训练模型）。你在终端机上对代码和数据的修改可以自动同步到远程机，在实际使用的过程中如同在远程机上编写程序一般，与串流游戏有异曲同工之妙。不过远程调试对网络的稳定性要求高，如果需要长时间训练模型，建议登录远程机终端直接训练模型（Linux 下可以结合 `nohup` 命令 [1](https://tf.wiki/zh/basic/installation.html#nohup) ，让进程在后端运行，不受终端退出的影响）。远程调试功能的具体配置步骤见 [PyCharm 文档](https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html) 。

**小技巧**

如果你是学生并有.edu 结尾的邮箱的话，可以在 [这里](http://www.jetbrains.com/student/) 申请 PyCharm 的免费 Professional 版本授权。

对于 TensorFlow 及深度学习的业余爱好者或者初学者， [Visual Studio Code](https://code.visualstudio.com/) 或者一些在线的交互式 Python 环境（比如免费的 [Google Colab](https://colab.research.google.com/) ）也是不错的选择。Colab 的使用方式可参考 [附录](https://tf.wiki/en/appendix/cloud.html#colab) 。

## TensorFlow 所需的硬件配置 

- 对于 TensorFlow 初学者，无需硬件升级也可以很好地学习和掌握 TensorFlow。如果自己的个人电脑难以胜任，可以考虑在云端（例如 [免费的 Colab](https://tf.wiki/en/appendix/cloud.html#colab) ）进行模型训练。
- 对于参加数据科学竞赛（比如 Kaggle）或者经常在本机进行训练的个人爱好者或开发者，一块高性能的 NVIDIA GPU 往往是必要的。CUDA 核心数和显存大小是决定显卡机器学习性能的两个关键参数，前者决定训练速度，后者决定可以训练多大的模型以及训练时的最大 Batch Size，对于较大规模的训练而言尤其敏感。
- 对于前沿的机器学习研究（尤其是计算机视觉和自然语言处理领域），多 GPU 并行训练是标准配置。为了快速迭代实验结果以及训练更大规模的

关于深度学习工作站的具体配置，由于硬件行情更新较快，故不在此列出具体配置，推荐关注 [知乎问题 - 如何配置一台适用于深度学习的工作站？](https://www.zhihu.com/question/33996159) ，并结合最新市场情况进行配置。