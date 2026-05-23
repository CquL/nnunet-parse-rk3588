# PARSE 模型从 `.pth` 转 `.rknn` 的流程

本文档只记录本次 PARSE nnUNetv2 模型如何从 PyTorch 权重转成 RK3588 可运行的 RKNN 文件。

## 1. 转换链路

本次模型转换不是直接：

```text
.pth -> .rknn
```

而是：

```text
checkpoint_best.pth
  -> 用 nnUNetv2/PyTorch 重建网络
  -> 导出 ONNX
  -> 用 RKNN Toolkit2 编译成 RKNN
```

也就是：

```text
.pth -> .onnx -> .rknn
```

## 2. 为什么不能只拿 `.pth` 直接转

`checkpoint_best.pth` 主要保存的是训练好的权重参数，不是一个完整的可部署系统。

nnUNetv2 的网络结构还需要结合这些文件恢复：

```text
dataset.json
plans.json
checkpoint_best.pth
```

本次模型目录是：

```text
D:\nnunet_parse_run\nnunetv2_PARSE_model_minimal\Dataset501_PARSE\nnUNetTrainer__nnUNetPlans__3d_fullres
```

其中：

```text
fold_0\checkpoint_best.pth
```

是权重文件。

所以第一步必须通过 nnUNetv2 代码把网络结构重建出来，并加载这个 checkpoint。

## 3. 第一步：`.pth` 导出 `.onnx`

导出脚本是：

```text
D:\nnunet_parse_run\export_nnunet_patch_onnx.py
```

这个脚本做了几件事：

```text
1. 设置 nnUNet_results / nnUNet_raw / nnUNet_preprocessed 环境变量
2. 用 nnUNetPredictor 读取训练好的模型文件夹
3. 加载 fold_0/checkpoint_best.pth
4. 得到 nnUNetv2 的 PyTorch 网络
5. 构造一个固定大小的 dummy input
6. 用 torch.onnx.export 导出 ONNX
```

核心代码逻辑是：

```python
predictor.initialize_from_trained_model_folder(
    str(MODEL_FOLDER),
    use_folds=(0,),
    checkpoint_name="checkpoint_best.pth",
)

network = predictor.network

torch.onnx.export(
    network,
    dummy_input,
    output_onnx,
    input_names=["image_patch"],
    output_names=["logits"],
    opset_version=19,
)
```

### 导出原版 patch ONNX

原版 nnUNetv2 patch 大小是：

```text
96 x 160 x 160
```

导出命令：

```powershell
cd D:\nnunet_parse_run
conda run -n d2l_gpu python scripts/export_nnunet_patch_onnx.py --patch-size 96x160x160 --output deployment\parse_3d_fullres_patch_96x160x160.onnx
```

得到：

```text
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_96x160x160.onnx
```

这个 ONNX 的输入输出是：

```text
input:  1 x 1 x 96 x 160 x 160
output: 1 x 2 x 96 x 160 x 160
```

### 导出当前 RKNN 验证版 ONNX

因为原版 `96x160x160` 在转 RKNN 时内存压力太大，所以当前先导出小 patch：

```text
32 x 64 x 64
```

导出命令：

```powershell
cd D:\nnunet_parse_run
conda run -n d2l_gpu python scripts/export_nnunet_patch_onnx.py --patch-size 32x64x64 --output deployment\parse_3d_fullres_patch_32x64x64.onnx
```

得到：

```text
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_32x64x64.onnx
```

这个 ONNX 的输入输出是：

```text
input:  1 x 1 x 32 x 64 x 64
output: 1 x 2 x 32 x 64 x 64
```

## 4. 第二步：`.onnx` 转 `.rknn`

ONNX 转 RKNN 的脚本是：

```text
D:\nnunet_parse_run\convert_onnx_to_rknn.py
```

这个脚本做了几件事：

```text
1. 创建 RKNN 对象
2. 设置 target_platform="rk3588"
3. 加载 ONNX
4. build 编译
5. 导出 .rknn
```

核心代码逻辑是：

```python
rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588")
rknn.load_onnx(model=str(onnx_model))
rknn.build(do_quantization=False)
rknn.export_rknn(str(rknn_model))
```

这里使用的是 FP 模式：

```text
do_quantization=False
```

也就是没有做 INT8 量化。

## 5. 实际成功转出的 RKNN

当前已经成功转出的 RKNN 是：

```text
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_32x64x64.rknn
```

使用的 ONNX 是：

```text
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_32x64x64.onnx
```

转换命令为：

```bash
cd /mnt/d/nnunet_parse_run
python scripts/convert_onnx_to_rknn.py --onnx deployment/parse_3d_fullres_patch_32x64x64.onnx --rknn deployment/parse_3d_fullres_patch_32x64x64.rknn
```

如果在 Windows PowerShell 中运行，需要保证当前环境安装了 RKNN Toolkit2。实际更常见是在 Linux/WSL 环境里运行 RKNN Toolkit2。

## 6. 为什么没有直接得到 `96x160x160.rknn`

`96x160x160` 的 ONNX 已经成功导出：

```text
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_96x160x160.onnx
```

但是转 RKNN 时占用内存过高，当时在 WSL 中编译被系统杀掉。

所以当前先使用：

```text
32x64x64
```

作为 RK3588 NPU 可行性验证版本。

这不代表原版不能做，而是说明下一步需要更大的编译内存、更合适的 RKNN 编译环境，或者进一步拆分/优化模型。

## 7. 当前转换产物

当前主要转换产物：

```text
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_96x160x160.onnx
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_32x64x64.onnx
D:\nnunet_parse_run\deployment\parse_3d_fullres_patch_32x64x64.rknn
```

文件含义：

```text
parse_3d_fullres_patch_96x160x160.onnx
  原版 nnUNetv2 patch 尺寸的 ONNX，中间格式。

parse_3d_fullres_patch_32x64x64.onnx
  当前 RKNN 验证版使用的小 patch ONNX。

parse_3d_fullres_patch_32x64x64.rknn
  当前已经部署到 RK3588 并跑通的 NPU 模型文件。
```

## 8. 参数量

两个 ONNX 的参数量相同：

```text
32x64x64 ONNX 参数量：30,784,450
96x160x160 ONNX 参数量：30,784,450
```

原因是 patch 尺寸只影响输入体素数量、计算量和内存占用，不改变网络卷积核数量。

## 9. 一句话总结

本次转换过程是：

```text
先用 nnUNetv2 读取模型文件夹和 checkpoint_best.pth，重建 PyTorch 网络；
再用 torch.onnx.export 导出固定 patch 输入的 ONNX；
最后用 RKNN Toolkit2 按 rk3588 平台把 ONNX 编译成 .rknn。
```

当前实际跑通的是：

```text
checkpoint_best.pth -> parse_3d_fullres_patch_32x64x64.onnx -> parse_3d_fullres_patch_32x64x64.rknn
```

