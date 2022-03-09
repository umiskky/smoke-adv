class Shape:
    def __init__(self) -> None:
        self.shape_input = [1, 640, 960, 3]
        self.shape_k = [1, 3, 3]
        self.shape_ratio = [1, 2]
        self.shape_box3d_branch = [1, 1, 50, 14]
        self.shape_feat = [1, 64, 160, 240]
