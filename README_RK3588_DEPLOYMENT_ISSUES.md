# RK3588 部署问题与解决办法

本文档记录本次 PARSE nnUNetv2 / RKNN 模型部署到 RK3588 设备时遇到的问题，以及对应解决办法。

## 1. `.rknn` 不能直接输入 `.nii.gz`

问题表现：

```text
以为 parse_3d_fullres_patch_32x64x64.rknn 可以像 nnUNetv2_predict 一样直接读取 .nii.gz。
```

原因：

```text
.rknn 只包含神经网络计算图和权重，只接收 tensor。
它不包含读取 NIfTI、归一化、切 patch、滑窗拼接、写回 .nii.gz 的逻辑。
```

解决办法：

```text
编写 device_infer_nii_rknn.py，在设备端完成：
.nii.gz -> numpy tensor -> RKNN/NPU 推理 -> mask tensor -> .nii.gz
```

## 2. 设备缺少 numpy / pip / rknnlite

问题表现：

```text
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'rknnlite'
```

原因：

```text
RK3588 系统自带 Python 环境不完整。
apt 安装 python3-pip 时还出现依赖问题。
```

解决办法：

先安装 numpy：

```bash
sudo apt-get install -y python3-numpy
```

如果系统没有 pip，用 `get-pip.py` 安装：

```bash
python3 get-pip.py --user
```

再安装 RKNNLite：

```bash
python3 -m pip install --user rknn-toolkit-lite2==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```bash
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "from rknnlite.api import RKNNLite; print('rknnlite OK')"
```

## 3. RKNN runtime 版本太旧

问题表现：

```text
Invalid RKNN model version 6
RKNN_ERR_MODEL_INVALID
librknnrt version: 1.5.0
```

原因：

```text
当前 .rknn 是 RKNN Toolkit2 2.3.2 编译的新模型格式。
设备上的 librknnrt.so 是 1.5.0，版本太旧，无法识别 model version 6。
```

解决办法：

先备份旧 runtime：

```bash
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak_1_5_0
```

替换为 2.3.2 runtime：

```bash
sudo cp rknnrt_2_3_2/librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig
```

验证：

```bash
python3 device_probe_rknn.py
```

成功时能看到：

```text
librknnrt version: 2.3.2
RKNN probe finished successfully.
```

如果替换后要回退：

```bash
sudo cp /usr/lib/librknnrt.so.bak_1_5_0 /usr/lib/librknnrt.so
sudo ldconfig
```

## 4. 设备下载 GitHub raw 文件很慢

问题表现：

```text
wget/curl 下载 raw.githubusercontent.com 的 librknnrt.so 时一直卡住。
```

原因：

```text
设备网络访问 GitHub raw 不稳定。
```

解决办法：

```text
先在 Windows 上下载 librknnrt.so，再通过 Xftp/ToDesk 传到 RK3588。
librknnrt.so 只有约 7.4MB，比直接让设备下载更稳定。
```

## 5. ToDesk 传输大文件太慢

问题表现：

```text
ToDesk 文件传输几百 MB 的部署包时速度为 0B/s 或非常慢。
```

原因：

```text
ToDesk 文件传输可能走中转，几百 MB 文件不稳定。
```

解决办法：

```text
优先使用 Xftp/SFTP。
如果必须远程传输，尽量只传最小部署包或单个小文件。
```

推荐传输方式：

```text
Windows: Xftp
协议: SFTP
端口: 22
用户: linaro
目标目录: /home/linaro/nnunet_parse
```

## 6. Python 3.8 不支持 `tuple[int, int, int]`

问题表现：

```text
TypeError: 'type' object is not subscriptable
```

原因：

```text
RK3588 设备上的 Python 是 3.8。
脚本里使用了 Python 3.9+ 的类型注解写法，例如 tuple[int, int, int]。
```

解决办法：

在 `device_infer_nii_rknn.py` 文件第一行加入：

```python
from __future__ import annotations
```

验证：

```bash
head -n 3 device_infer_nii_rknn.py
```

应该看到：

```text
from __future__ import annotations
```

## 7. 多行命令被拆成多个命令

问题表现：

```text
bash: -i: command not found
bash: -o: command not found
bash: --center-patch-only: command not found
```

原因：

```text
复制多行命令时，反斜杠 \ 后面多了空行，导致下一行被当成新的命令。
```

解决办法：

```text
新手操作尽量使用一整行命令。
```

例如：

```bash
python3 device_infer_nii_rknn.py -i ../PA000005_0000.nii.gz -o ../PA000005_rknn_mask.nii.gz
```

## 8. 96x160x160 原版 patch 转 RKNN 内存不够

问题表现：

```text
96x160x160 ONNX 可以导出，但转 RKNN 时编译进程被系统杀掉。
```

原因：

```text
RKNN 编译阶段内存占用过高，当前 WSL/机器内存不足。
```

解决办法：

```text
先使用 32x64x64 小 patch 版本完成 RK3588 NPU 可行性验证。
后续正式版再尝试更大内存机器、更合适的 RKNN 编译环境，或者优化模型/切分方式。
```

## 9. 当前只看到一个 NPU core 有负载

问题表现：

```text
NPU load: Core0: 63%, Core1: 0%, Core2: 0%
```

说明：

```text
模型确实在 NPU 上跑，但当前主要使用 Core0。
```

后续优化方向：

```text
尝试 RKNNLite.NPU_CORE_0_1_2；
多 RKNN 实例并行处理 patch；
重新研究 RKNN 编译和多核调度配置。
```

## 10. 如何确认最终跑通

先测试 RKNN 能否加载和推理：

```bash
python3 device_probe_rknn.py
```

成功标志：

```text
RKNN probe finished successfully.
```

再跑 `.nii.gz` 完整推理：

```bash
python3 device_infer_nii_rknn.py -i ../PA000005_0000.nii.gz -o ../PA000005_rknn_mask.nii.gz
```

成功输出：

```text
../PA000005_rknn_mask.nii.gz
```

监控 NPU：

```bash
watch -n 1 "free -h; echo; ps -o pid,cmd,%mem,rss,vsz -C python3; echo; sudo cat /sys/kernel/debug/rknpu/load 2>/dev/null"
```

