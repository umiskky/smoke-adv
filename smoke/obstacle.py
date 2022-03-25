import torch

from smoke.utils import AffineUtils


class Obstacle:
    type_map = {0: "CAR", 1: "CYCLIST", 2: "WALKER",
                3: "Van", 4: "Truck", 5: "Person_sitting",
                6: "Tram", 7: "Misc", 8: "DontCare"}
    type_map_inv = {"CAR": 0, "CYCLIST": 1, "PEDESTRIAN": 2,
                    "Van": 3, "Truck": 4, "Person_sitting": 5,
                    "Tram": 6, "Misc": 7, "DontCare": 8}
    occluded_map = {0: "fully visible", 1: "partly occluded",
                    2: "largely occluded", 3: "unknown"}
    occluded_map_inv = {"fully visible": 0, "partly occluded": 1,
                        "largely occluded": 2, "unknown": 3}

    def __init__(self) -> None:
        self.type = None
        self.truncated = None
        self.occluded = None
        self.alpha = None
        self.box2d = None
        self.dimensions = None
        self.location = None
        self.rotation_y = None
        self.score = None
        self.box3d = None

    @staticmethod
    def decode(box3d_branch_data: torch.Tensor, k: torch.Tensor, confidence_score: float, ori_img_size: list) -> list:
        """
        解析smoke模型输出box3d_branch，返回检测对象的list.\n
        :param k: camera intrinsic matrix, can be found in obstacle object.
        :param confidence_score: confidence threshold.
        :param box3d_branch_data: smoke box3d_branch output in torch.Tensor type.
        :param ori_img_size: raw image size. [h, w]
        :return: list of obstacles.
        """
        # move tensor to cpu
        device = torch.device("cpu")
        box3d_branch_data = box3d_branch_data.cpu() if not box3d_branch_data.device == device else box3d_branch_data
        k = k.cpu() if not k.device == device else k

        obstacle_list = []
        with torch.no_grad():
            total_pred = box3d_branch_data

            # filter box using confidence_threshold
            score = confidence_score
            keep_idx = torch.nonzero(total_pred[:, -1] > score).squeeze()
            total_pred = total_pred.index_select(0, keep_idx)

            if total_pred.shape[0] > 0:
                # alpha_z
                pred_alpha = total_pred[:, 1]
                # [h, l, w] -> [w, h, l]
                pred_dimensions = total_pred[:, 6:9].roll(shifts=1, dims=1)
                # [x, y, z]
                pred_locations = total_pred[:, 9:12]
                # pred_alpha -> pred_rotation_y
                pred_rotation_y = AffineUtils.alpha2rotation_y_N(pred_alpha, pred_locations[:, 0], pred_locations[:, 2])
                # shape=[10, 2, 8]
                box3d_image = AffineUtils.recovery_3d_box(pred_rotation_y, pred_dimensions, pred_locations, k, ori_img_size)

                for idx in range(total_pred.shape[0]):
                    item = total_pred[idx, :]
                    obstacle = Obstacle()
                    obstacle.type = item[0].int()
                    obstacle.dimensions = pred_dimensions[idx].numpy().tolist()
                    obstacle.location = pred_locations[idx].numpy().tolist()
                    obstacle.rotation_y = pred_rotation_y[idx].numpy().tolist()
                    obstacle.box3d = box3d_image[idx].numpy().tolist()
                    obstacle.box2d = item[2:6].numpy().tolist()
                    obstacle.score = item[-1].item()
                    obstacle_list.append(obstacle)

            return obstacle_list
