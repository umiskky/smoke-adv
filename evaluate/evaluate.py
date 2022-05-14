"""This script is used to evaluate."""
import argparse
import copy
import os
import os.path as osp
import pickle
import shutil
import sys
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from tools.file_utils import makedirs
from dataset.metric import Metric
from pipeline.pipeline import Pipeline
from tools.config import Config
from tools.logger import Logger

warnings.filterwarnings("ignore")


def evaluate(pipeline: Pipeline, dataset: dict, cfg: Config, off_dir: str, prefix) -> dict:
    step = 0
    metrics_dict = {}
    pipeline.visualization.reset(off_dir)
    t_angle = tqdm(total=len(dataset.keys()))
    for angle, samples in dataset.items():
        t_scenario = tqdm(total=len(samples))
        # init for metrics dict
        if metrics_dict.get(angle) is None:
            metrics_dict[angle] = {}
        for sample in samples:
            metric = Metric()
            # Record GT
            metric.sample = sample
            # inference forward
            box3d_branch, box_pseudo_gt = pipeline.forward(sample)
            # filter
            box3d_branch_3d_radius_filtered = filter_with_3d_radius(box3d_branch=box3d_branch,
                                                                    box_3d_gt=copy.deepcopy(box_pseudo_gt.get("3d")),
                                                                    radius=2)
            box3d_branch_filter = box3d_branch_3d_radius_filtered
            # Record GT
            metric.dimension_gt = cfg.cfg_object.get("size")
            metric.box_2d_gt = box_pseudo_gt.get("2d")
            # select prediction with max score
            if box3d_branch_filter is not None and box3d_branch_filter.shape[0] > 0:
                # sort detection results by score
                _, indices = torch.sort(box3d_branch_filter[:, -1], dim=0, descending=True, stable=True)
                indices = torch.flatten(indices)
                # Get max score item
                prediction = box3d_branch_filter[indices[0], :]
                # Record Prediction
                metric.class_type = prediction[0].int().item()
                metric.score = prediction[-1].item()
                metric.dimension_pred = prediction[6:9].roll(shifts=1, dims=0).cpu().numpy().tolist()
                location = prediction[9:12]
                location[1] -= float(cfg.cfg_renderer["camera"]["height"])
                metric.location_pred = location.cpu().numpy().tolist()
                metric.box_2d_pred = prediction[2:6].cpu().numpy().tolist()
                # Set metric status
                if metric.score < 0.25:
                    metric.status = 2
                elif metric.class_type != 2:
                    metric.status = 1
                else:
                    metric.status = 0
            else:
                metric.status = 3
            metrics_dict[angle][sample.scenario_index] = metric
            t_scenario.update()
            step += 1
            # Vis
            pipeline.visualization.vis(scenario_index=sample.scenario_index,
                                       scenario=pipeline.scenario,
                                       renderer=pipeline.renderer,
                                       stickers=pipeline.stickers,
                                       defense=pipeline.defense,
                                       smoke=pipeline.smoke,
                                       prefix=prefix,
                                       suffix="_angle-%s" % str(angle))
        t_scenario.close()
        t_angle.update()
    t_angle.close()
    return metrics_dict


def saving_metrics(metrics_dict: dict, fp):
    with open(fp, 'wb') as fpb:
        pickle.dump(metrics_dict, fpb)


def reading_metrics(fp) -> dict:
    with open(fp, 'rb') as fpb:
        metrics_dict: dict = pickle.load(fpb)
    return metrics_dict


def filter_with_3d_radius(box3d_branch: torch.Tensor, box_3d_gt: dict, radius: float = 3.0):
    """filter box3d_branch with 3d location which is in a circle of radius"""
    if box3d_branch is None:
        return None
    if radius < 0:
        return box3d_branch
    device = box3d_branch.device
    box_3d_gt_location = box_3d_gt.get('location')
    h_offset = box_3d_gt.get('h_offset')
    if h_offset is not None:
        box_3d_gt_location[1] += h_offset
    if isinstance(box_3d_gt_location, list):
        box_3d_gt_location = torch.tensor(box_3d_gt_location, device=device)
    elif isinstance(box_3d_gt_location, torch.Tensor):
        box_3d_gt_location = box_3d_gt_location.to(device)
    location = box3d_branch[:, 9:12]
    distances = F.pairwise_distance(location, box_3d_gt_location, p=2)
    keep_idx = torch.flatten(torch.nonzero(distances.le(radius)))
    if keep_idx.shape[0] <= 0:
        return None
    box3d_branch_ = box3d_branch.index_select(0, keep_idx)
    return box3d_branch_


def eval_norm(pipeline: Pipeline, fp: str):
    metrics = []
    # 0~1.0 RGB HWC float32
    texture_raw = pipeline.object_loader.textures.clone().squeeze().cpu()
    # 0~1.0 RGB HWC float32
    texture_perturb = pipeline.stickers.visualization.get("texture_perturb")
    if texture_perturb is not None:
        texture_perturb = texture_perturb.cpu()
        # 0~1.0 RGB HWC -> CHW float32
        d_texture = (texture_perturb - texture_raw).permute(2, 0, 1)

        # L1 norm
        rgb_l1_norm = torch.mean(torch.norm(d_texture, p=1, dim=0))
        metrics.append(rgb_l1_norm.cpu().item())

        # frobenius norm
        rgb_frobenius_norm = torch.mean(torch.norm(d_texture, p='fro', dim=0))
        metrics.append(rgb_frobenius_norm.cpu().item())

        # inf norm
        rgb_inf_norm = torch.mean(torch.norm(d_texture, p=float('inf'), dim=0))
        metrics.append(rgb_inf_norm.cpu().item())

        df = pd.DataFrame([metrics], columns=["rgb_l1_norm", "rgb_frobenius_norm", "rgb_inf_norm"])
        print(df)
        df.to_csv(fp)


def eval_attack_success_rate(metrics_dict: dict, fp: str, raw_metrics_dict=None):
    # Count the num of different status in metrics
    index = []
    data = []
    for angle, metrics in metrics_dict.items():
        index.append(angle)
        normal_num = 0
        class_num = 0
        score_num = 0
        location_num = 0
        for metric in metrics.values():
            if metric.status == 0:
                normal_num += 1
            elif metric.status == 1:
                class_num += 1
            elif metric.status == 2:
                score_num += 1
            elif metric.status == 3:
                location_num += 1
            else:
                raise ValueError("Invalid metric status for %d." % metric.status)
        data.append([normal_num, class_num, score_num, location_num, normal_num + class_num + score_num + location_num])
    df = pd.DataFrame(data, index=index, columns=["normal", "class", "score", "location", "total"])
    df_angle_sum = pd.DataFrame([df.sum()], index=["Global"])
    df = pd.concat([df, df_angle_sum])

    # Calculate attack success rate in different status
    if raw_metrics_dict:
        index = []
        data = []
        for angle in metrics_dict.keys():
            raw_metrics = raw_metrics_dict.get(angle)
            metrics = metrics_dict.get(angle)
            index.append(angle)
            d = 0
            # s1=[s, c, n]->[l]  s2=[c, n]->[l, s]  s3=[n]->[l, s, c]
            s1 = 0
            s2 = 0
            s3 = 0
            raw_normal_num = 0
            raw_class_num = 0
            raw_score_num = 0
            raw_location_num = 0

            for scenario in metrics.keys():
                metric = metrics.get(scenario)
                raw_metric = raw_metrics.get(scenario)

                if metric.status in [3] and raw_metric.status in [0, 1, 2]:
                    s1 += 1
                if metric.status in [2, 3] and raw_metric.status in [0, 1]:
                    s2 += 1
                if metric.status in [1, 2, 3] and raw_metric.status in [0]:
                    s3 += 1
                if metric.status == 0:
                    d += 1

                if raw_metric.status == 0:
                    raw_normal_num += 1
                elif raw_metric.status == 1:
                    raw_class_num += 1
                elif raw_metric.status == 2:
                    raw_score_num += 1
                elif raw_metric.status == 3:
                    raw_location_num += 1
                else:
                    raise ValueError("Invalid raw_metric status for %d." % raw_metric.status)
            data.append([d, s1, s2, s3,
                         raw_normal_num + raw_class_num + raw_score_num + raw_location_num,
                         raw_normal_num + raw_class_num + raw_score_num,
                         raw_normal_num + raw_class_num,
                         raw_normal_num])
        df_tmp = pd.DataFrame(data, index=index,
                              columns=["d", "s1", "s2", "s3",
                                       "total_d", "total_s1", "total_s2", "total_s3"])
        df_tmp_angle_sum = pd.DataFrame([df_tmp.sum()], index=["Global"])
        df_tmp = pd.concat([df_tmp, df_tmp_angle_sum])
        d = df_tmp["d"] / df_tmp["total_d"]
        s1 = df_tmp["s1"] / df_tmp["total_s1"]
        s2 = df_tmp["s2"] / df_tmp["total_s2"]
        s3 = df_tmp["s3"] / df_tmp["total_s3"]
    else:
        d = df["normal"] / df["total"]
        s1 = df["location"] / df["total"]
        s2 = (df["location"] + df["score"]) / df["total"]
        s3 = (df["location"] + df["score"] + df['class']) / df["total"]
    s_df = pd.DataFrame({'d': d, 's1': s1, 's2': s2, 's3': s3})
    df = pd.concat([df, s_df], axis=1)
    # Verbose
    print(df)
    df.to_csv(fp)


def main_pipe(args):
    cfg = Config(args.cfg)
    logger = Logger(cfg.cfg_logger)
    logger.broadcast_logger(cfg.cfg_all, exclude=[])
    pipeline = Pipeline(cfg)
    dataset = pipeline.dataset.eval_data

    global_path = cfg.cfg_global["project_path"]
    timestamp = os.getenv("timestamp")

    # mode
    meta: str = cfg.cfg_dataset["meta"]
    meta_name = (meta.split("/")[-1]).split(".")[0]
    mode = "train" if meta_name == "meta_train" else "eval"

    # =========================== Solve Defense Saving Path ==========================
    sub_dir = ""
    if cfg.cfg_enable["defense"]:
        purifier_type = cfg.cfg_defense["type"]
        if "GaussianBlur" == purifier_type:
            gb_params = cfg.cfg_defense["gaussian_blur"]
            sub_dir = "gaussian_blur/kernel_{}___sigma_{}".format(gb_params["kernel_size"], gb_params["sigma"])
        elif "MedianBlur" == purifier_type:
            mb_params = cfg.cfg_defense["median_blur"]
            sub_dir = "median_blur/kernel_{}".format(mb_params["kernel_size"])
        elif "BitDepth" == purifier_type:
            bd_params = cfg.cfg_defense["bit_depth"]
            sub_dir = "bit_depth/r_{}___g_{}___b_{}".format(bd_params["r_bits"],
                                                            bd_params["g_bits"],
                                                            bd_params["b_bits"])
        elif "JpegCompression" == purifier_type:
            jc_params = cfg.cfg_defense["jpeg_compression"]
            sub_dir = "jpeg_compression/quality_{}".format(jc_params["quality"])

    # =================================== Raw Eval ===================================
    raw_fd = osp.join(global_path, cfg.cfg_eval["raw_eval_path"], sub_dir)
    makedirs(raw_fd)
    raw_fp = osp.join(raw_fd, "raw_%s_eval.pickle" % mode)
    if osp.exists(raw_fp):
        raw_eval_metrics_dict = reading_metrics(raw_fp)
        print("Raw %s dataset evaluation already exists." % mode)
    else:
        print("Evaluate raw %s dataset ..." % mode)
        raw_eval_metrics_dict = evaluate(pipeline=pipeline, dataset=copy.deepcopy(dataset),
                                         cfg=cfg, off_dir=osp.join(raw_fd, mode, "visualization"),
                                         prefix="")
        saving_metrics(raw_eval_metrics_dict, raw_fp)
    eval_attack_success_rate(raw_eval_metrics_dict, osp.join(raw_fd, "%s_eval.csv" % mode))

    # =============================== Dataset Raw Eval ===============================
    if cfg.cfg_enable["defense"]:
        raw_fd = osp.join(global_path, cfg.cfg_eval["dataset_raw_path"])
        makedirs(raw_fd)
        raw_fp = osp.join(raw_fd, "raw_%s_eval.pickle" % mode)
        if osp.exists(raw_fp):
            raw_eval_metrics_dict = reading_metrics(raw_fp)
            print("Raw %s dataset evaluation already exists." % mode)
        else:
            print("Evaluate raw %s dataset ..." % mode)
            raw_eval_metrics_dict = evaluate(pipeline=pipeline, dataset=copy.deepcopy(dataset),
                                             cfg=cfg, off_dir=osp.join(raw_fd, mode, "visualization"),
                                             prefix="")
            saving_metrics(raw_eval_metrics_dict, raw_fp)
        eval_attack_success_rate(raw_eval_metrics_dict, osp.join(raw_fd, "%s_eval.csv" % mode))

    # ================================= Attack Eval ==================================
    attack_fd = osp.join(global_path, cfg.cfg_eval["attack_eval_path"], sub_dir, mode + "_" + timestamp)
    makedirs(attack_fd)
    attack_fp = osp.join(attack_fd, "attack_%s_eval.pickle" % mode)
    texture_hls_fp = osp.join(global_path, cfg.cfg_eval["texture_hls_path"])
    if not osp.exists(texture_hls_fp):
        print("Adversarial texture file not exists!")
        sys.exit(0)
    else:
        # Copy texture file to eval dir.
        shutil.copy(texture_hls_fp, osp.join(attack_fd, "adv_texture.pth"))
        pipeline.stickers.adv_texture_hls = texture_hls_fp
        if osp.exists(attack_fp):
            attack_eval_metrics_dict = reading_metrics(attack_fp)
        else:
            attack_eval_metrics_dict = evaluate(pipeline=pipeline, dataset=copy.deepcopy(dataset),
                                                cfg=cfg, off_dir=osp.join(attack_fd, "visualization"),
                                                prefix="")
            saving_metrics(attack_eval_metrics_dict, attack_fp)
    eval_norm(pipeline, osp.join(attack_fd, "%s_norm.csv" % mode))
    eval_attack_success_rate(attack_eval_metrics_dict, osp.join(attack_fd, "%s_eval.csv" % mode), raw_eval_metrics_dict)

    # =================================== GN Eval ====================================
    if cfg.cfg_eval["gn_enable"]:
        gn_fd = osp.join(global_path, cfg.cfg_eval["gn_eval_path"], sub_dir, mode + "_" + timestamp)
        makedirs(gn_fd)
        gn_fp = osp.join(gn_fd, "gn_%s_eval.pickle" % mode)
        # Generate gn perturb
        pipeline.stickers.apply_gauss_perturb(cfg.cfg_attack["optimizer"]["clip_min"] * 3,
                                              cfg.cfg_attack["optimizer"]["clip_max"] * 3)
        gn_eval_metrics_dict = evaluate(pipeline=pipeline, dataset=copy.deepcopy(dataset),
                                        cfg=cfg, off_dir=osp.join(gn_fd, "visualization"),
                                        prefix="")
        saving_metrics(gn_eval_metrics_dict, gn_fp)
        eval_norm(pipeline, osp.join(gn_fd, "%s_norm.csv" % mode))
        eval_attack_success_rate(gn_eval_metrics_dict, osp.join(gn_fd, "%s_eval.csv" % mode), raw_eval_metrics_dict)


def parse_args():
    parser = argparse.ArgumentParser(description='Attack Pipeline')

    # params of evaluate
    parser.add_argument(
        "--config",
        "-f",
        dest="cfg",
        default="../data/config/defense.yaml",
        help="The config file path.",
        required=False,
        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main_pipe(args)