# Smoke-ADV 

---
### 1. Smoke Model Loader & Extractor

---
- Model Input
  - Smoke模型输入有三个，分别为大小为960*640的图像tensor、相机的内参矩阵K、
    以及图片缩放率down_ratio(建议是整数，即960\*640的整数倍)。
- Model Output
  - ![img.png](data/docs/image/output.png)
  - class 为识别的类别：0: "CAR", 1: "CYCLIST", 2: "PEDESTRIAN"
  - alpha 对应绕相机y轴的旋转角，对应alpha_z
  - ~~2D Box Coordinate 为2D Box坐标~~
  - 3D Box Dimension 对应3D Box的h、l、w
  - 3D Box Center Coordinate 对应车底盘中心点在相机坐标系下的坐标，注意不是车体中心点
  - score 对应识别物体的置信度

### 2. Kitti/Waymo Dataset Scenarios

---
- Note
- kitti dataset
  - setting
    - ![img.png](data/docs/image/img.png)
  - kitti scenarios
    - ![000011](data/docs/image/000011.png)
    - ![000062](data/docs/image/000062.png)
    - ![000448](data/docs/image/000448.png)
- waymo dataset
  - setting
  404 NOT FOUND
  - waymo scenarios
    - ![0002095](data/docs/image/0002095.png)

### 3. Differentiable Rendering

---
- Note
- Coordinate System Define
  -  ![coordinate_systems](data/docs/image/coordinate_systems.png)
  - World Coordinate
    - 因为Waymo Open Dataset暂时找不到传感器之间详细的位置关系，所以暂时参考kitti设置
    - ```
      # from rear to front
      #        ^ y    
      #        |       
      #      z ⊙--> x
      ```
  - Camera Coordinate
    - ```
      # from rear to front
      #        ^ y
      #        |
      #  x <-- ⊕ z
      ```
  - Image Coordinate
- Pytorch 3D
  - ![](data/docs/image/architecture_renderer.jpg)
  - ![](data/docs/image/transforms_overview.jpg)

### 4. Attack

---


### 5. Defense

---

# TODO
- ransac 外参标定
- 确定3D模型所在位置，造假的GT，能不能渲染模型到真人身后
- 考虑攻击分类器置信度之外，3D Box回归框进行攻击，比如让预测的回归框大小变化
- 24号字
- 光照问题，可以通过图像匹配算法，风格。。。
- 定位问题
  - 3D GT IOU 定位
  - MASK 2D 定位
  - 中心点定位