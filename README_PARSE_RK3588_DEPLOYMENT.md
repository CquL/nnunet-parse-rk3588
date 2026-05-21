# PARSE nnUNetv2 到 RK3588 NPU 部署说明

本文档记录本次 PARSE nnUNetv2 模型从 Windows 本地推理，到 ONNX/RKNN 转换，再到 RK3588 设备推理验证的流程。

## 1. 原始文件

一开始只有两个压缩包：

```text
D:\Desktop\旷枢平台功能介绍\nnunetv2_PARSE_model_minimal.zip
D:\Desktop\旷枢平台功能介绍\nnunetv2_PARSE_test_images.zip
```

解压/整理后的主要工作目录是：

```text
D:\nnunet_parse_run
```

其中：

```text
D:\nnunet_parse_run\nnunetv2_PARSE_model_minimal
```

是 nnUNetv2 模型文件夹，里面包含：

```text
Dataset501_PARSE/
  nnUNetTrainer__nnUNetPlans__3d_fullres/
    dataset.json
    plans.json
    fold_0/
      checkpoint_best.pth
```

`checkpoint_best.pth` 是训练好的权重文件。

测试数据在：

```text
D:\nnunet_parse_run\nnunetv2_PARSE_test_images
```

里面有：

```text
PA000005_0000.nii.gz
PA000016_0000.nii.gz
```

## 2. Windows 本地 nnUNetv2 推理

Windows 上使用 conda 环境：

```text
d2l_gpu
```

nnUNetv2 原版推理会直接读取 `.nii.gz`，自动做预处理、滑窗推理、融合，然后输出 `.nii.gz` mask。

运行脚本：

```text
D:\nnunet_parse_run\run_inference_here.ps1
```

输出目录：

```text
D:\nnunet_parse_run\predictions
```

输出结果示例：

```text
D:\nnunet_parse_run\predictions\PA000005.nii.gz
D:\nnunet_parse_run\predictions\PA000016.nii.gz
```

## 3. 模型转换路线

RK3588 的 NPU 不能直接运行 `.pth`，所以转换路线是：

```text
checkpoint_best.pth
  -> 用 nnUNetv2/PyTorch 重建网络并加载权重
  -> 导出 ONNX
  -> 用 RKNN Toolkit2 转成 RKNN
  -> RK3588 设备加载 .rknn 推理
```

也就是：

```text
.pth -> .onnx -> .rknn
```

当前转换产物在：

```text
D:\nnunet_parse_run\deployment
```

主要文件：

```text
parse_3d_fullres_patch_32x64x64.onnx
parse_3d_fullres_patch_96x160x160.onnx
parse_3d_fullres_patch_32x64x64.rknn
```

说明：

```text
96x160x160 ONNX 可以导出，但当时转 RKNN 时内存不够。
32x64x64 RKNN 已经成功生成，并已在 RK3588 上跑通。
```

## 4. 当前 RKNN 模型

当前真正部署到 RK3588 的文件是：

```text
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_32x64x64.rknn
```

它的输入 shape 是：

```text
1 x 1 x 32 x 64 x 64
```

含义：

```text
batch = 1
channel = 1
Z = 32
Y = 64
X = 64
```

输出 shape 是：

```text
1 x 2 x 32 x 64 x 64
```

其中 `2` 表示两个类别：

```text
0 = background
1 = vessel
```

## 5. 参数量统计

用 ONNX 统计得到当前部署网络参数量：

```text
30,784,450
```

也就是约：

```text
30.78M parameters
```

注意：

```text
32x64x64 版本参数量 = 30,784,450
96x160x160 版本参数量 = 30,784,450
```

因为输入 patch 大小变了，但网络卷积核数量没有变，所以参数量一样。输入变大主要影响计算量、内存占用和单次看到的空间范围。

统计脚本：

```text
D:\Desktop\旷枢平台功能介绍\count_model_params.py
```

运行示例：

```powershell
conda run -n d2l_gpu python "D:\Desktop\旷枢平台功能介绍\count_model_params.py" "D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_32x64x64.onnx"
```

## 6. RK3588 设备端文件

给设备用的最小部署包是：

```text
D:\nnunet_parse_run_github_private\artifacts\rk3588_parse_min_deploy_package.zip
```

设备上解压到：

```text
/home/linaro/nnunet_parse/rk3588_parse_min_deploy
```

设备端主要文件：

```text
/home/linaro/nnunet_parse/rk3588_parse_min_deploy/PA000005_0000.nii.gz
/home/linaro/nnunet_parse/rk3588_parse_min_deploy/deployment/parse_3d_fullres_patch_32x64x64.rknn
/home/linaro/nnunet_parse/rk3588_parse_min_deploy/deployment/device_probe_rknn.py
/home/linaro/nnunet_parse/rk3588_parse_min_deploy/deployment/device_infer_nii_rknn.py
```

其中：

```text
device_probe_rknn.py
```

用于测试 `.rknn` 能不能在 RK3588 NPU 上加载和推理。

```text
device_infer_nii_rknn.py
```

用于读取 `.nii.gz`，切 patch，调用 RKNN/NPU 推理，再拼回 `.nii.gz` 输出。

## 7. RK3588 上实际运行命令

进入目录：

```bash
cd /home/linaro/nnunet_parse/rk3588_parse_min_deploy/deployment
```

测试 RKNN 能不能跑：

```bash
python3 device_probe_rknn.py
```

成功日志里可以看到：

```text
librknnrt version: 2.3.2
target platform: rk3588
RKNN probe finished successfully.
```

完整 `.nii.gz` 推理：

```bash
python3 device_infer_nii_rknn.py -i ../PA000005_0000.nii.gz -o ../PA000005_rknn_mask.nii.gz
```

输出结果：

```text
/home/linaro/nnunet_parse/rk3588_parse_min_deploy/PA000005_rknn_mask.nii.gz
```

## 8. RKNN Runtime 版本问题

设备原本的 RKNN runtime 是：

```text
librknnrt version: 1.5.0
```

会报错：

```text
Invalid RKNN model version 6
```

原因是当前 `.rknn` 是 RKNN Toolkit2 2.3.2 编译出来的新格式，旧 runtime 不能识别。

后来把设备 runtime 替换为：

```text
librknnrt version: 2.3.2
```

之后模型成功运行。

备份旧 runtime：

```bash
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak_1_5_0
```

替换为新 runtime：

```bash
sudo cp rknnrt_2_3_2/librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig
```

## 9. 如何看 NPU/内存占用

在 RK3588 上新开终端执行：

```bash
watch -n 1 "free -h; echo; ps -o pid,cmd,%mem,rss,vsz -C python3; echo; sudo cat /sys/kernel/debug/rknpu/load 2>/dev/null"
```

其中：

```text
free -h      看整机内存
RSS          看 python 推理进程实际内存占用
rknpu/load   看 NPU core 负载
```

实际测试中看到：

```text
NPU load: Core0: 63%, Core1: 0%, Core2: 0%
```

说明推理确实跑在 RK3588 NPU 上，但当前实际主要使用 Core0。

## 10. 如何查看分割结果

RK3588 输出的是 `.nii.gz`，不是普通图片。需要传回 Windows 后导出 PNG 预览。

设备输出文件传回 Windows：

```text
D:\nnunet_parse_run_github_private\artifacts\PA000005_rknn_mask.nii.gz
```

导出三栏对比图脚本：

```text
D:\Desktop\旷枢平台功能介绍\export_rk3588_compare.py
```

运行：

```powershell
conda run -n d2l_gpu python "D:\Desktop\旷枢平台功能介绍\export_rk3588_compare.py" --image "D:\nnunet_parse_run\nnunetv2_PARSE_test_images\PA000005_0000.nii.gz" --mask "D:\nnunet_parse_run_github_private\artifacts\PA000005_rknn_mask.nii.gz" --output "D:\nnunet_parse_run\rk3588_full_compare_png" --slices 12
```

输出目录：

```text
D:\nnunet_parse_run\rk3588_full_compare_png
```

每张图是三栏：

```text
左边：原始 CT
中间：RK3588 输出 mask
右边：红色叠加结果
```

## 11. 当前效果对比

对 `PA000005`，Windows 原版 nnUNetv2 输出和 RK3588 输出粗略对比：

```text
原版 nnUNetv2 前景体素：262,404
RK3588 RKNN 前景体素：196,572
Dice：0.533431
IoU：0.363727
```

说明：

```text
RK3588 NPU 已经能跑通，并且能输出分割结果。
但当前版本还不是完整等价的 nnUNetv2。
```

## 12. 当前版本和真正 nnUNetv2 的区别

当前 RK3588 版本是：

```text
nnUNetv2 的神经网络核心 + 简化版前后处理 + RKNN/NPU 部署
```

真正 nnUNetv2 包含：

```text
读取 .nii.gz
重采样 spacing
完整 CT 归一化
96x160x160 patch
滑窗推理
Gaussian 加权融合
镜像 TTA
后处理
写回 .nii.gz
```

当前 RK3588 版本：

```text
读取 .nii.gz
简化 CT 归一化
32x64x64 patch
普通平均融合
无 TTA
写回 .nii.gz
```

所以当前版本定位是：

```text
RK3588 NPU 可行性验证版
```

还不是：

```text
完全等价 nnUNetv2 部署版
```

## 13. 下一步如果要做成正式版

建议继续做：

```text
1. 尝试把 96x160x160 patch 的 ONNX 成功转成 RKNN。
2. 复刻 nnUNetv2 原版 preprocessing/resampling。
3. 实现 Gaussian 滑窗融合。
4. 加镜像 TTA。
5. 对多个病例和 Windows 原版 nnUNetv2 做 Dice/IoU 对比。
6. 再考虑多 NPU core 并行和部署包加密。
```

## 14. 部署过程中遇到的问题和解决办法

### 14.1 nnUNetv2 找不到模型目录

现象：

```text
nnUNet_results is not defined
Could not find a dataset with the ID 501
```

原因：

```text
nnUNetv2 推理依赖 nnUNet_results、nnUNet_raw、nnUNet_preprocessed 环境变量。
如果不设置，nnUNetv2 不知道模型和数据在哪里。
```

解决：

```text
把环境变量写进 run_inference_here.ps1 / run_inference_here.bat / run_inference_here.py，
以后通过脚本运行，不手动重复设置。
```

### 14.2 中文路径导致读取医学图像不稳定

现象：

```text
SimpleITK/nnUNetv2 在中文路径下读取 .nii.gz 容易出问题。
```

解决：

```text
把模型和测试数据复制到英文路径：
D:\nnunet_parse_run
```

### 14.3 `.rknn` 不能直接读取 `.nii.gz`

现象：

```text
以为 .rknn 可以像 nnUNetv2_predict 一样直接输入 .nii.gz。
```

原因：

```text
.rknn 只包含神经网络计算图和权重，只接收 tensor。
它不包含读 NIfTI、归一化、切 patch、拼回 mask 的流程。
```

解决：

```text
编写 device_infer_nii_rknn.py：
.nii.gz -> numpy tensor -> RKNN/NPU -> mask tensor -> .nii.gz
```

### 14.4 原始 96x160x160 patch 转 RKNN 时内存不够

现象：

```text
96x160x160 ONNX 可以导出，但转 RKNN 时编译进程被系统杀掉。
```

原因：

```text
RKNN 编译阶段内存占用过高，当前 WSL/机器内存不足。
```

解决：

```text
先导出并转换 32x64x64 小 patch 版本，作为 RK3588 NPU 可行性验证。
后续正式版再尝试更大内存环境或优化转换 96x160x160。
```

### 14.5 ToDesk 传输大文件太慢

现象：

```text
ToDesk 文件传输几百 MB 包时速度为 0B/s 或很慢。
```

解决：

```text
优先用 Xftp/SFTP 传输；
或者只传最小部署包；
也可以在 Windows 上临时开 HTTP 服务，让 RK3588 用 wget 下载。
```

### 14.6 RK3588 上缺少 numpy / pip / rknnlite

现象：

```text
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'rknnlite'
```

原因：

```text
设备系统 Python 环境不完整，apt 安装 python3-pip 时还遇到依赖问题。
```

解决：

```bash
sudo apt-get install -y python3-numpy
python3 get-pip.py --user
python3 -m pip install --user rknn-toolkit-lite2==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 14.7 RKNN runtime 版本太旧

现象：

```text
Invalid RKNN model version 6
RKNN_ERR_MODEL_INVALID
librknnrt version: 1.5.0
```

原因：

```text
当前 .rknn 是 RKNN Toolkit2 2.3.2 编译出来的，
但设备上的 librknnrt.so 是 1.5.0，旧 runtime 不认识新模型格式。
```

解决：

```bash
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak_1_5_0
sudo cp rknnrt_2_3_2/librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig
```

验证：

```text
librknnrt version: 2.3.2
RKNN probe finished successfully.
```

### 14.8 设备访问 GitHub raw 很慢

现象：

```text
wget/curl 下载 raw.githubusercontent.com 文件时一直卡住。
```

解决：

```text
在 Windows 上先下载 librknnrt.so，再通过 Xftp/ToDesk 传到设备。
因为 librknnrt.so 只有约 7.4MB，比传整个模型包稳定。
```

### 14.9 Python 3.8 不支持 `tuple[int, int, int]`

现象：

```text
TypeError: 'type' object is not subscriptable
```

原因：

```text
RK3588 设备 Python 是 3.8，脚本里用了 Python 3.9+ 类型写法。
```

解决：

```python
from __future__ import annotations
```

把这行加到 `device_infer_nii_rknn.py` 文件最上面。

### 14.10 多行命令被拆成多个命令

现象：

```text
bash: -i: command not found
bash: -o: command not found
```

原因：

```text
复制多行命令时，反斜杠 \ 后面多了空行或没有正确续行。
```

解决：

```text
给新手操作时尽量使用一整行命令。
```

例如：

```bash
python3 device_infer_nii_rknn.py -i ../PA000005_0000.nii.gz -o ../PA000005_rknn_mask.nii.gz
```

### 14.11 当前只看到一个 NPU core 有负载

现象：

```text
NPU load: Core0: 63%, Core1: 0%, Core2: 0%
```

说明：

```text
当前模型确实在 NPU 上跑，但主要使用 Core0。
```

后续优化方向：

```text
尝试 RKNNLite.NPU_CORE_0_1_2；
或者多 RKNN 实例并行处理 patch；
或者重新研究 RKNN 编译和多核调度配置。
```
