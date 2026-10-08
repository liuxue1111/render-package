# RadarSplat Renderer Demo

一个完全独立的 RadarSplat 距离—方位角（RA）渲染教学项目。使用者只需修改
`configs/example.yaml`，即可观察场景、RA 功率图和占用图的变化。

它不导入 `radarsplat_rio`，不包含训练、建图或数据集代码；底层计算仍使用
RadarSplat 官方 CUDA rasterizer。

## 目录

```text
radarsplat-renderer-demo/
├── configs/example.yaml   # 主要修改这里
├── demo.py                # 独立渲染和可视化入口
├── outputs/               # 自动生成的结果
├── environment.yml
├── requirements.txt
└── README.md
```

## 使用

先复制一份配置：

```bash
cp configs/example.yaml configs/my_experiment.yaml
```

运行：

```bash
python demo.py \
  --config configs/my_experiment.yaml \
  --output outputs/my_experiment
```

输出包括：

- `render.png`：场景俯视图、RA 功率图、RA 占用图；
- `render_arrays.npz`：功率、占用、距离和方位角数组；
- `summary.json`：输出尺寸和可见目标。

## 可以修改什么

| 参数 | 含义 | 预期现象 |
| --- | --- | --- |
| `position_m` | 目标坐标 `[x,y,z]` | 改变峰值距离和方位角 |
| `scale_m` | Gaussian 三轴尺度 | 改变回波扩散范围 |
| `opacity` | 不透明度 `(0,1)` | 改变目标贡献 |
| `reflectance` | 反射强度 | 改变功率亮度 |
| `radar.position_m` | 雷达位置 | 改变相对位置 |
| `radar.yaw_deg` | 雷达偏航角 | 改变目标方位角 |
| `azimuth_beamwidth_deg` | 方位波束宽度 | 改变方位模糊程度 |
| `spectral_leakage_enabled` | 频谱泄漏开关 | 改变距离方向扩散 |

四元数顺序为 `w,x,y,z`；角度单位为度，距离单位为米。

## 环境安装

要求 Linux、NVIDIA GPU 和 CUDA。推荐 Python 3.9、PyTorch 2.1.2、CUDA 11.8。
官方渲染器不能在纯 CPU 环境运行。

```bash
conda env create -f environment.yml
conda activate radarsplat-renderer-demo

python -m pip install \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

git clone --recursive https://github.com/umautobots/radarsplat.git third_party/radarsplat
git -C third_party/radarsplat checkout ea9c8f530c708622cc3b1b560436b5557ac6a49b
python -m pip install ninja jaxtyping rich
python -m pip install -e third_party/radarsplat \
  --no-build-isolation --config-settings editable_mode=compat

python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python demo.py
```

CUDA 检查应输出 `True`。

## 推荐实验

1. 把目标从 `[6,0,0]` 移到 `[9,0,0]`，观察距离峰值。
2. 把目标移到 `[6,3,0]`，观察方位角。
3. 单独改变 `reflectance`，比较功率与几何位置。
4. 增大 `scale_m`，观察响应面积。
5. 开关 `spectral_leakage_enabled`。
6. 改变雷达位置和 `yaw_deg`。

## 坐标约定

- 场景位于世界坐标系；
- Demo 先构造雷达到世界的 `T_WR`，渲染时使用 `T_RW = inverse(T_WR)`；
- 原始 RA 轴顺序为 `[azimuth, range, channel]`；
- 数值方位角为 `[0,360)`，仅绘图时重排成 `[-180,180)`。

## 上传 GitHub

本目录拥有独立 Git 仓库，不会进入 `radarsplat-rio` 的版本历史。创建 GitHub 空仓库后：

```bash
git add .
git commit -m "Initial RadarSplat renderer teaching demo"
git remote add origin <你的 GitHub 仓库链接>
git push -u origin main
```

`.gitignore` 已排除 `third_party/` 和生成的 `outputs/`。

底层 CUDA 实现来自 [UM AutoBots RadarSplat](https://github.com/umautobots/radarsplat)。
归属与版本见 `NOTICE.md`，许可见 `LICENSE`。
