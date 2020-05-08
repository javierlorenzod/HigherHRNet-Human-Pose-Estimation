# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import numpy
import json
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torchvision.datasets
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

import dataset_jld as dsjld

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')
from loguru import logger as loggur

class ToNumpy(object):
    def __call__(self, image):
        return numpy.array(image)

def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()
    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    transforms_pre = torchvision.transforms.Compose(
        [
            ToNumpy(),
        ]
    )
    # iterate over all datasets
    datasets_root_path = "/media/jld/DATOS_JLD/datasets"
    datasets = ["cityscapes", "kitti", "tsinghua"]
    # testing sets from cityscapes and kitti does not have groundtruth --> processing not required
    datasplits = [["train", "val"], ["train"], ["train", "val", "test"]]
    keypoints_output_root_path = "/media/jld/DATOS_JLD/git-repos/paper-revista-keypoints/results"
    model_name = osp.basename(cfg.TEST.MODEL_FILE).split('.')[0] # Model name + configuration
    for dsid, dataset in enumerate(datasets):
        dataset_root_path = osp.join(datasets_root_path, dataset)
        output_root_path = osp.join(keypoints_output_root_path, dataset)
        for datasplit in datasplits[dsid]:
            loggur.info(f"Processing split {datasplit} of {dataset}")
            input_img_dir = osp.join(dataset_root_path, datasplit)
            output_kps_json_dir = osp.join(output_root_path, datasplit, model_name)
            loggur.info(f"Input image dir: {input_img_dir}")
            loggur.info(f"Output pose JSON dir: {output_kps_json_dir}")
            # test_dataset = torchvision.datasets.ImageFolder("/media/jld/DATOS_JLD/git-repos/paper-revista-keypoints/test_images/", transform=transforms_pre)
            test_dataset = dsjld.BaseDataset(input_img_dir,
                                             output_kps_json_dir,
                                             transform=transforms_pre)
            test_dataset.generate_io_samples_pairs()
            # Stablish weight of keypoints scores (like openpifpaf in https://github.com/vita-epfl/openpifpaf/blob/master/openpifpaf/decoder/annotation.py#L44)
            n_keypoints = 17
            kps_score_weights = numpy.ones((17,))
            kps_score_weights[:3] = 3.0
            # Normalize weights to sum 1
            kps_score_weights /= numpy.sum(kps_score_weights)
            data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            parser = HeatmapParser(cfg)
            all_preds = []
            all_scores = []

            pbar = tqdm(total=len(test_dataset)) # if cfg.TEST.LOG_PROGRESS else None
            for i, (img, imgidx) in enumerate(data_loader):
                assert 1 == img.size(0), 'Test batch size should be 1'

                img = img[0].cpu().numpy()
                # size at scale 1.0
                base_size, center, scale = get_multi_scale_size(
                    img, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
                )

                with torch.no_grad():
                    final_heatmaps = None
                    tags_list = []
                    for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                        input_size = cfg.DATASET.INPUT_SIZE
                        image_resized, center, scale = resize_align_multi_scale(
                            img, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                        )
                        image_resized = transforms(image_resized)
                        image_resized = image_resized.unsqueeze(0).cuda()

                        outputs, heatmaps, tags = get_multi_stage_outputs(
                            cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                            cfg.TEST.PROJECT2IMAGE, base_size
                        )

                        final_heatmaps, tags_list = aggregate_results(
                            cfg, s, final_heatmaps, tags_list, heatmaps, tags
                        )

                    final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
                    tags = torch.cat(tags_list, dim=4)
                    grouped, scores = parser.parse(
                        final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                    )

                    final_results = get_final_preds(
                        grouped, center, scale,
                        [final_heatmaps.size(3), final_heatmaps.size(2)]
                    )

                # if cfg.TEST.LOG_PROGRESS:
                pbar.update()
                # Save all keypoints in a JSON dict
                final_json_results = []
                for kps in final_results:
                    kpsdict = {}
                    x = kps[:, 0]
                    y = kps[:, 1]
                    kps_scores = kps[:, 2]
                    kpsdict['keypoints'] = kps[:, 0:3].tolist()
                    # bounding box by means of minmax approach (without zero elements)
                    xmin = numpy.float64(numpy.min(x[numpy.nonzero(x)]))
                    xmax = numpy.float64(numpy.max(x))
                    width = numpy.float64(xmax - xmin)
                    ymin = numpy.float64(numpy.min(y[numpy.nonzero(y)]))
                    ymax = numpy.float64(numpy.max(y))
                    height = numpy.float64(ymax - ymin)
                    kpsdict['bbox'] = [xmin, ymin, width, height]
                    # Calculate pose score as a weighted mean of keypoints scores
                    kpsdict['score'] = numpy.float64(numpy.sum(kps_score_weights * numpy.sort(kps_scores)[::-1]))
                    final_json_results.append(kpsdict)

                with open(test_dataset.output_json_files_list[imgidx], "w") as f:
                    json.dump(final_json_results, f)

                all_preds.append(final_results)
                all_scores.append(scores)

            if cfg.TEST.LOG_PROGRESS:
                pbar.close()
"""
    name_values, _ = test_dataset.evaluate(
        cfg, all_preds, all_scores, final_output_dir
    )

    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME)
"""

if __name__ == '__main__':
    main()
