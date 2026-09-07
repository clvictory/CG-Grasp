import os

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps


def save_clean_results(
    rgb_img,
    grasp_q_img,
    grasp_angle_img,
    depth_img=None,
    no_grasps=1,
    grasp_width_img=None,
    save_dir='results',
    grasp_color='#32CD32'
):
    """Save evaluation outputs without matplotlib frames."""
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    os.makedirs(save_dir, exist_ok=True)
    colorbar_dir = os.path.join(save_dir, 'with_colorbar')
    heatmap_dir = os.path.join(save_dir, 'heatmap_only')
    os.makedirs(colorbar_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    rgb_path = os.path.join(save_dir, 'rgb.png')
    if np.issubdtype(rgb_img.dtype, np.floating):
        plt.imsave(rgb_path, np.clip(rgb_img, 0.0, 1.0))
    else:
        plt.imsave(rgb_path, rgb_img)

    if depth_img is not None and depth_img.size:
        depth_path = os.path.join(save_dir, 'depth.png')
        if np.issubdtype(depth_img.dtype, np.floating):
            depth_to_save = np.clip(depth_img, 0.0, 1.0)
        else:
            depth_to_save = depth_img
        plt.imsave(depth_path, depth_to_save, cmap='gray')

    height, width = rgb_img.shape[0], rgb_img.shape[1]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax, color=grasp_color)
    ax.axis('off')
    fig.savefig(os.path.join(save_dir, 'grasp.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig(os.path.join(colorbar_dir, 'quality.png'))
    plt.close(fig)

    plt.imsave(os.path.join(heatmap_dir, 'quality.png'), grasp_q_img, cmap='jet', vmin=0, vmax=1)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig(os.path.join(colorbar_dir, 'angle.png'))
    plt.close(fig)

    plt.imsave(os.path.join(heatmap_dir, 'angle.png'), grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig(os.path.join(colorbar_dir, 'width.png'))
    plt.close(fig)

    plt.imsave(os.path.join(heatmap_dir, 'width.png'), grasp_width_img, cmap='jet', vmin=0, vmax=100)
