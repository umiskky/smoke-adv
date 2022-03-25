import torch


class AffineUtils:

    @staticmethod
    def alpha2rotation_y_N(alpha_N: torch.Tensor, x_N: torch.Tensor, z_N: torch.Tensor):
        """
        矩阵计算绕摄像机y轴旋转角[-PI, PI]. \n
        :param alpha_N: alpha in shape N
        :param x_N: object center coordinate x in Camera Coordinate System in shape N
        :param z_N: object center coordinate y in Camera Coordinate System in shape N
        :return: Tensor
        """
        # 以摄像机坐标系z轴为横轴, x为纵轴; ry[-PI, PI]且顺时针旋转为正;
        # atan(x/z)对应视角的ry_view; alpha(ry_object)为alpha_z, 以视角向量为横轴, 顺时针旋转为正;
        # 考虑两次偏航旋转ry = ry_view + ry_object.
        rotation_y_N = alpha_N + torch.atan2(x_N, z_N)
        less_index = torch.nonzero(rotation_y_N < -torch.pi).squeeze()
        large_index = torch.nonzero(rotation_y_N > torch.pi).squeeze()
        rotation_y_N[less_index] += 2 * torch.pi
        rotation_y_N[large_index] -= 2 * torch.pi
        return rotation_y_N

    @staticmethod
    def rad_to_matrix(rotys, N):
        """
        获得shape为N的沿y轴的旋转矩阵 \n
        :param rotys: rotation angle along y axis.
        :param N: shape N.
        :return: Tensor
        """
        cos, sin = rotys.cos(), rotys.sin()
        i_temp = torch.FloatTensor([[1, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 1]])
        # i_temp复制N份, ry = N * i_temp
        # [3, 3] -> [3*N, 3] -> [N, 3, 3]
        ry = torch.reshape(i_temp.tile((N, 1)), (N, -1, 3))
        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos
        return ry

    @staticmethod
    def recovery_3d_box(rotys, dims, locs, K, image_size):
        """
        计算每个物体的3D Box. \n
        :param rotys: rotation y. [N]
        :param dims: box dimensions. [N, 3]
        :param locs: box center point coordinate. [n, 3]
        :param K: camera intrinsic matrix.
        :param image_size: raw image plane size.
        :return: box3d in image plane. Tensor
        """
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()
        if len(dims.shape) == 3:
            dims = torch.reshape(dims, (-1, 3))
        if len(locs.shape) == 3:
            locs = torch.reshape(locs, (-1, 3))

        N = rotys.shape[0]
        ry = AffineUtils.rad_to_matrix(rotys, N)
        dims = torch.reshape(dims, (-1, 1)).tile((1, 8))
        # x, z
        dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
        dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
        # 是-1，说明预测的不是车辆正中心的点，而是车辆底盘正中心的点
        # y
        dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
        # [3, 8] -> [3*N, 8]
        index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                              [4, 5, 0, 1, 6, 7, 2, 3],
                              [4, 5, 6, 0, 1, 2, 3, 7]]).tile((N, 1))
        box_3d_object = torch.gather(dims, 1, index)
        # 对所有坐标点进行变换
        box_3d = torch.matmul(ry, torch.reshape(box_3d_object, (N, 3, -1)))
        # locs在最后一个维度增加一维[N, 3] -> [N, 3, 1] -> [N, 3, 8]
        box_3d += locs.unsqueeze(-1).tile((1, 1, 8))
        # 相机内参矩阵K[1, 3, 3]
        box3d_image = torch.matmul(K, box_3d)
        # 把z轴数据除掉，获得x，y坐标数据[N, 2, 8]
        box3d_image = box3d_image[:, :2, :] / torch.reshape(box3d_image[:, 2, :], (box_3d.shape[0], 1, box_3d.shape[2]))
        # 先取整，再转化为浮点数类型
        box3d_image = box3d_image.int()
        box3d_image = box3d_image.float()
        # 将box坐标限制在图片大小范围内
        box3d_image[:, 0] = box3d_image[:, 0].clip(0, image_size[1])
        box3d_image[:, 1] = box3d_image[:, 1].clip(0, image_size[0])
        return box3d_image
