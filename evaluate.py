import argparse
import logging
import time

import numpy as np
import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')
    parser.add_argument('--out-dir', default='', type=str)

    # Network
    parser.add_argument('--network', metavar='N', type=str, nargs='+',
                        default="",
                        help='Path to saved networks to evaluate')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")',
                        default='graspnet')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset',
                        default='/home/datasets/GraspNet_1B/')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=True,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', action='store_true', default=True,
                        help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true',
                        help='Jacquard-dataset style output')

    # Misc.
    parser.add_argument('--vis', action='store_true',default=False,
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(42)  # 设置全局随机种子42

    # Get the compute device
    torch.cuda.set_device('cuda:1')
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path,
                           output_size=args.input_size,
                           ds_rotate=args.ds_rotate,
                           random_rotate=args.augment,
                           random_zoom=args.augment,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb)

    indices = list(range(test_dataset.length))
    split = int(np.floor(args.split * test_dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)

    # start = int(0.2 * test_dataset.length)
    # end = int(0.3 * test_dataset.length)
    # val_indices = indices[start:end]
    
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info('Validation size: {}'.format(len(val_indices)))

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # for network in args.network:
    logging.info('\nEvaluating model {}'.format(args.network))

    # Load Network
    net = torch.load(args.network, map_location=device,weights_only = False)

    results = {'correct': 0, 'failed': 0}

     # 新增：用于统计误差的累积列表
    angle_errors = []
    center_offsets = []
    width_errors = []

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    start_time = time.time()

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                    no_grasps=args.n_grasps,
                                                    grasp_width=width_img,
                                                    threshold=args.iou_threshold
                                                    )
                if s:
                    results['correct'] += 1

                    # ===== 新增：提取预测抓取并计算误差 =====
                    # 获取最佳预测抓取（只取一个）
                    pred_grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                    if pred_grasps:
                        pred_g = pred_grasps[0]  # Grasp对象
                        gt_rects = test_data.dataset.get_gtbb(didx, rot, zoom)

                        # 寻找与预测抓取最匹配的真实框（使用中心距离最小作为近似）
                        best_gt = None
                        min_dist = float('inf')
                        for gt_rect in gt_rects:
                            # GraspRectangle对象通常有center属性
                            gt_center = gt_rect.center
                            pred_center = pred_g.center
                            dist = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
                            if dist < min_dist:
                                min_dist = dist
                                best_gt = gt_rect

                        if best_gt is not None:
                            # 获取真实抓取参数
                            gt_center = best_gt.center
                            gt_angle = best_gt.angle
                            gt_width = best_gt.width

                            # 预测抓取参数
                            pred_center = pred_g.center
                            pred_angle = pred_g.angle
                            pred_width = pred_g.width

                            # 角度误差（处理周期性，得到最小绝对差）
                            angle_diff = np.abs(np.arctan2(np.sin(pred_angle - gt_angle),
                                                           np.cos(pred_angle - gt_angle)))
                            # 中心偏移（欧氏距离，单位像素）
                            center_offset = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
                            # 宽度偏差（绝对差）
                            width_diff = np.abs(pred_width - gt_width)

                            angle_errors.append(angle_diff)
                            center_offsets.append(center_offset)
                            width_errors.append(width_diff)
                    
                else:
                    results['failed'] += 1

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.vis:
                save_results(
                    rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                    depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=args.n_grasps,
                    grasp_width_img=width_img,
                    id=idx,
                    ds=args.dataset,
                    out_dir=args.out_dir,
                )

    avg_time = (time.time() - start_time) / len(test_data)
    logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                    results['correct'] + results['failed'],
                                                    results['correct'] / (results['correct'] + results['failed'])))

    # 打印平均误差指标（仅在存在成功样本时）
    if len(angle_errors) > 0:
        logging.info('Average Angle Error: {:.4f} rad ({:.2f} deg)'.format(
            np.mean(angle_errors), np.mean(angle_errors) * 180 / np.pi))
        logging.info('Average Center Offset: {:.2f} px'.format(np.mean(center_offsets)))
        logging.info('Average Width Error: {:.2f} px'.format(np.mean(width_errors)))
    else:
        logging.info('No successful grasps to compute error metrics.')

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))

    del net
    torch.cuda.empty_cache()