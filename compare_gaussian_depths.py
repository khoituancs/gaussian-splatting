import torch
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import json

# --- XỬ LÝ ĐƯỜNG DẪN ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_DIR = os.path.join(SCRIPT_DIR, 'Depth-Anything-V2')

if not os.path.isdir(DEPTH_ANYTHING_DIR):
    print(f"Lỗi: Không tìm thấy thư mục 'Depth-Anything-V2' tại '{DEPTH_ANYTHING_DIR}'.")
    sys.exit(1)

if DEPTH_ANYTHING_DIR not in sys.path:
    sys.path.append(DEPTH_ANYTHING_DIR)

# Import các thành phần cần thiết
from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, get_combined_args
from utils.general_utils import safe_state
from torchvision.transforms import Compose
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
except ImportError as e:
    print(f"Lỗi: Không thể import module từ Depth Anything V2. Chi tiết: {e}")
    sys.exit(1)

def run_depth_anything(image_path, model, transform):
    raw_image = cv2.imread(image_path)
    if raw_image is None: return None, (0, 0)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    with torch.no_grad():
        depth = model(image)
    return depth.squeeze().cpu().numpy(), (h, w)

def render_gaussian_center_depth(view, gaussians):
    xyz = gaussians.get_xyz
    view_matrix = view.world_view_transform.cuda()
    xyz_h = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device="cuda")], dim=1)
    p_view = xyz_h @ view_matrix
    return p_view[:, 2], xyz

def calculate_depth_loss(gauss_depth_map, estimated_depth_map):
    """Tính toán loss (MAE) giữa hai bản đồ độ sâu đã chuẩn hóa."""
    if gauss_depth_map.shape != estimated_depth_map.shape:
        estimated_depth_map = cv2.resize(estimated_depth_map, (gauss_depth_map.shape[1], gauss_depth_map.shape[0]))
    
    g_min, g_max = gauss_depth_map.min(), gauss_depth_map.max()
    e_min, e_max = estimated_depth_map.min(), estimated_depth_map.max()

    gauss_norm = (gauss_depth_map - g_min) / (g_max - g_min) if g_max - g_min > 0 else np.zeros_like(gauss_depth_map)
    est_norm = (estimated_depth_map - e_min) / (e_max - e_min) if e_max - e_min > 0 else np.zeros_like(estimated_depth_map)
        
    # Tính Mean Absolute Error (MAE)
    loss = np.mean(np.abs(gauss_norm - est_norm))
    
    # Trả về các bản đồ đã chuẩn hóa để trực quan hóa và giá trị loss
    return gauss_norm, est_norm, loss

def visualize_and_save(gauss_norm, est_norm, output_path):
    """Tạo và lưu ảnh so sánh trực quan."""
    diff_map = np.abs(gauss_norm - est_norm)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    axes[0].imshow(1 - gauss_norm, cmap='viridis'); axes[0].set_title('Depth từ Tâm Gaussian'); axes[0].axis('off')
    axes[1].imshow(est_norm, cmap='viridis'); axes[1].set_title('Depth từ Depth Anything V2'); axes[1].axis('off')
    axes[2].imshow(diff_map, cmap='inferno', vmin=0, vmax=1); axes[2].set_title('Sự khác biệt Tuyệt đối (Loss Map)'); axes[2].axis('off')

    plt.suptitle(f'So sánh Depth Map cho ảnh: {os.path.basename(output_path)}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def compare_all_depths(dataset: ModelParams, iteration: int):
    print("Đang thiết lập mô hình Depth Anything V2...")
    model_path = os.path.join(DEPTH_ANYTHING_DIR, 'checkpoints', 'depth_anything_v2_vitl.pth')
    try:
        model_configs = {'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
        depth_anything_model = DepthAnythingV2(**model_configs['vitl'])
        depth_anything_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        depth_anything_model = depth_anything_model.cuda().eval()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp trọng số tại '{model_path}'.")
        return
        
    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    with torch.no_grad():
        print("Đang tải mô hình 3D Gaussian Splatting...")
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        views = scene.getTestCameras() or scene.getTrainCameras()
        if not views:
            print("Lỗi: Không tìm thấy camera nào."); return

        output_dir = os.path.join(dataset.model_path, f"depth_comparison_{iteration}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Kết quả sẽ được lưu tại: {output_dir}")

        all_losses = {} # Khởi tạo dictionary để lưu loss

        for view in tqdm(views, desc="So sánh độ sâu trên tất cả các góc nhìn"):
            image_path = os.path.join(dataset.source_path, dataset.images, view.image_name)
            if not os.path.exists(image_path): continue
            
            estimated_depth, (h, w) = run_depth_anything(image_path, depth_anything_model, transform)
            if estimated_depth is None: continue
            estimated_depth = cv2.resize(estimated_depth, (w, h), interpolation=cv2.INTER_LINEAR)

            depths_3d, xyz_3d = render_gaussian_center_depth(view, gaussians)
            
            proj_matrix = view.full_proj_transform.cuda()
            image_height, image_width = view.image_height, view.image_width
            
            xyz_h = torch.cat([xyz_3d, torch.ones((xyz_3d.shape[0], 1), device="cuda")], dim=1)
            xyz_clip = xyz_h @ proj_matrix
            xyz_clip[:, :2] /= xyz_clip[:, 3, None]
            
            screen_x = (xyz_clip[:, 0] + 1) * image_width / 2
            screen_y = (xyz_clip[:, 1] + 1) * image_height / 2

            mask = (screen_x >= 0) & (screen_x < image_width) & (screen_y >= 0) & (screen_y < image_height) & (depths_3d > 0)
            
            screen_points = torch.stack([screen_x[mask], screen_y[mask]], dim=1).cpu().numpy().astype(np.int32)
            depth_values = depths_3d[mask].cpu().numpy()
            
            gauss_depth_map = np.full((image_height, image_width), np.inf, dtype=np.float32)
            for (px, py), depth_val in zip(screen_points, depth_values):
                if 0 <= py < image_height and 0 <= px < image_width and depth_val < gauss_depth_map[py, px]:
                    gauss_depth_map[py, px] = depth_val
            
            if np.any(np.isfinite(gauss_depth_map)):
                max_finite_depth = np.max(gauss_depth_map[np.isfinite(gauss_depth_map)])
                gauss_depth_map[np.isinf(gauss_depth_map)] = max_finite_depth
            else:
                gauss_depth_map.fill(0)

            # Tính toán loss và lấy các bản đồ đã chuẩn hóa
            gauss_norm, est_norm, loss_value = calculate_depth_loss(gauss_depth_map, estimated_depth)
            all_losses[view.image_name] = loss_value
            
            # Trực quan hóa và lưu ảnh so sánh
            output_path = os.path.join(output_dir, f"comparison_{os.path.basename(view.image_name)}.png")
            visualize_and_save(gauss_norm, est_norm, output_path)

        # Sau khi hoàn tất, tính loss trung bình và lưu kết quả
        avg_loss = np.mean(list(all_losses.values()))
        loss_output_path = os.path.join(output_dir, "depth_loss_results.txt")
        
        with open(loss_output_path, 'w', encoding='utf-8') as f:
            f.write(f"Kết quả so sánh Độ sâu (Mean Absolute Error) cho checkpoint: {iteration}\n")
            f.write("="*50 + "\n")
            f.write(f"Loss trung bình trên tất cả các góc nhìn: {avg_loss:.6f}\n")
            f.write("="*50 + "\n\n")
            f.write("Loss chi tiết cho từng ảnh:\n")
            # Sắp xếp loss từ cao đến thấp để dễ xem
            sorted_losses = sorted(all_losses.items(), key=lambda item: item[1], reverse=True)
            for name, loss in sorted_losses:
                f.write(f"- {name}: {loss:.6f}\n")
        
        print(f"\nHoàn tất! Đã lưu kết quả loss vào: {loss_output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Script so sánh độ sâu giữa 3DGS và Depth Anything V2.")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=30000, type=int, help="Vòng lặp checkpoint cần phân tích.")
    args = get_combined_args(parser)
    safe_state(True)
    compare_all_depths(model.extract(args), args.iteration)
