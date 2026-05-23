# 通用流程：`.pth/.pt` 转 `.rknn`

核心流程：

```text
.pth/.pt -> PyTorch 重建模型 -> 导出 .onnx -> RKNN Toolkit2 转 .rknn
```

也就是：

```text
PyTorch -> ONNX -> RKNN
```

## 环境配置

假设别人已经进入项目根目录：

```bash
cd <project-root>
```

环境说明文件：

```text
requirements_pth_to_rknn.txt
```

简化理解：

```text
1. 导出 ONNX 的环境：Python 3.11、torch、onnx、numpy、nnunetv2、SimpleITK
2. 转 RKNN 的环境：rknn-toolkit2==2.3.2，建议 Linux/WSL/Docker
3. RK3588 运行环境：rknn-toolkit-lite2==2.3.2 + 匹配的 librknnrt.so
```

本项目 ONNX 导出环境实测版本：

```text
Python 3.11.14
torch 2.12.0.dev20260222+cu128
onnx 1.21.0
numpy 2.4.2
nnunetv2 2.7.0
SimpleITK 2.5.5
```

`torch` 不建议死抄这个 dev 版本，别人机器上应安装和自己 CUDA/CPU 匹配的稳定 PyTorch 版本。

## 代码文件

### 1. 本项目：`checkpoint_best.pth -> .onnx`

本项目 PARSE/nnUNetv2 使用专用脚本：

```text
export_nnunet_patch_onnx.py
```

原因：

```text
nnUNetv2 不能只靠一个 checkpoint_best.pth 直接导出 ONNX。
它需要 dataset.json、plans.json、checkpoint_best.pth 和 nnUNetv2 目录结构一起恢复网络。
```

本项目实际 ONNX 导出命令：

```bash
python scripts/export_nnunet_patch_onnx.py --patch-size 32x64x64 --output deployment/parse_3d_fullres_patch_32x64x64.onnx
```

如果要导出原版 patch 尺寸：

```bash
python scripts/export_nnunet_patch_onnx.py --patch-size 96x160x160 --output deployment/parse_3d_fullres_patch_96x160x160.onnx
```

### 2. 普通 PyTorch 模型：`.pth/.pt -> .onnx`

普通 PyTorch 模型可以参考这个通用模板脚本：

```text
export_pytorch_to_onnx_general.py
```

这个脚本适合这类普通 PyTorch 模型：

```text
可以通过 module:Class 创建模型结构
可以通过 load_state_dict 加载 .pth/.pt 权重
输入 shape 是固定的
```

示例命令：

```bash
python export_pytorch_to_onnx_general.py --weights model.pth --model your_model_file:YourModel --input-shape 1x3x224x224 --output deployment/model.onnx
```

如果模型构造函数需要参数：

```bash
python export_pytorch_to_onnx_general.py --weights model.pth --model your_model_file:YourModel --model-kwargs-json '{"num_classes":2}' --input-shape 1x3x224x224 --output deployment/model.onnx
```

注意：

```text
这个脚本不是绝对万能。
如果模型是 nnUNetv2、YOLO、Detectron、MMDetection 等复杂框架模型，通常需要写专用导出脚本。
```

### 3. `.onnx -> .rknn`

这个脚本已经相对通用，可以复用：

```text
convert_onnx_to_rknn.py
```

示例命令：

```bash
python scripts/convert_onnx_to_rknn.py --onnx deployment/model.onnx --rknn deployment/model.rknn
```

本项目实际 RKNN 转换命令：

```bash
python scripts/convert_onnx_to_rknn.py --onnx deployment/parse_3d_fullres_patch_32x64x64.onnx --rknn deployment/parse_3d_fullres_patch_32x64x64.rknn
```

输出：

```text
deployment/parse_3d_fullres_patch_32x64x64.rknn
```

## 本项目结论

这次不是直接 `.pth -> .rknn`，而是写了两步代码：

```text
export_nnunet_patch_onnx.py     checkpoint_best.pth + nnUNetv2 结构 -> .onnx
convert_onnx_to_rknn.py         .onnx -> .rknn
```

已经跑通的实际链路：

```text
checkpoint_best.pth
  -> parse_3d_fullres_patch_32x64x64.onnx
  -> parse_3d_fullres_patch_32x64x64.rknn
```
