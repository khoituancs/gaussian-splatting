import torch
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# --- XỬ LÝ ĐƯỜNG DẪN ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_DIR = os.path.join(SCRIPT_DIR, 'Depth-Anything-V2')

if not os.path.isdir(DEPTH_ANYTHING_DIR):
    print(f"Lỗi: Không tìm thấy thư mục 'Depth-Anything-V2' tại '{DEPTH_ANYTHING_DIR}'.")
    sys.exit(1)

if DEPTH_ANYTHING_DIR not in sys.path:
    sys.path.append(DEPTH_ANYTHING_DIR)

# Import các thành phần cần thiết
from scene import Scene, GaussianModel
from gaussian_renderer import render, GaussianRasterizationSettings, GaussianRasterizer
from arguments import ModelParams, PipelineParams, get_combined_args
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

def calculate_depth_loss(rendered_depth_map, estimated_depth_map):
    """Tính toán loss (MAE) giữa hai bản đồ độ sâu đã chuẩn hóa."""
    if rendered_depth_map.shape != estimated_depth_map.shape:
        estimated_depth_map = cv2.resize(estimated_depth_map, (rendered_depth_map.shape[1], rendered_depth_map.shape[0]))
    
    r_min, r_max = rendered_depth_map.min(), rendered_depth_map.max()
    e_min, e_max = estimated_depth_map.min(), estimated_depth_map.max()

    rendered_norm = (rendered_depth_map - r_min) / (r_max - r_min) if r_max - r_min > 0 else np.zeros_like(rendered_depth_map)
    est_norm = (estimated_depth_map - e_min) / (e_max - e_min) if e_max - e_min > 0 else np.zeros_like(estimated_depth_map)
        
    loss = np.mean(np.abs(rendered_norm - est_norm))
    return rendered_norm, est_norm, loss

def visualize_and_save(rendered_norm, est_norm, output_path):
    """Tạo và lưu ảnh so sánh trực quan."""
    diff_map = np.abs(rendered_norm - est_norm)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    axes[0].imshow(rendered_norm, cmap='viridis'); axes[0].set_title('Depth kết xuất từ 3DGS'); axes[0].axis('off')
    axes[1].imshow(est_norm, cmap='viridis'); axes[1].set_title('Depth từ Depth Anything V2'); axes[1].axis('off')
    axes[2].imshow(diff_map, cmap='inferno', vmin=0, vmax=1); axes[2].set_title('Sự khác biệt Tuyệt đối (Loss Map)'); axes[2].axis('off')

    plt.suptitle(f'So sánh Depth Map cho ảnh: {os.path.basename(output_path)}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def compare_rendered_depths(dataset_args: ModelParams, pipeline_args: PipelineParams, iteration: int):
    """Hàm chính để thực hiện toàn bộ quy trình so sánh."""
    
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
        gaussians = GaussianModel(dataset_args.sh_degree)
        scene = Scene(dataset_args, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset_args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        views = scene.getTestCameras() or scene.getTrainCameras()
        if not views:
            print("Lỗi: Không tìm thấy camera nào."); return

        output_dir = os.path.join(dataset_args.model_path, f"rendered_depth_comparison_{iteration}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Kết quả sẽ được lưu tại: {output_dir}")

        all_losses = {} # Khởi tạo dictionary để lưu loss

        for view in tqdm(views, desc="So sánh độ sâu trên tất cả các góc nhìn"):
            # 1. Chạy Depth Anything
            image_path = os.path.join(dataset_args.source_path, dataset_args.images, view.image_name)
            if not os.path.exists(image_path): continue
            
            estimated_depth, (h, w) = run_depth_anything(image_path, depth_anything_model, transform)
            if estimated_depth is None: continue
            estimated_depth = cv2.resize(estimated_depth, (w, h), interpolation=cv2.INTER_LINEAR)

            # 2. Render độ sâu tổng hợp từ 3DGS
            # Dùng override_color để tăng tốc bằng cách bỏ qua tính toán màu sắc
            render_pkg = render(view, gaussians, pipeline_args, background, override_color=torch.zeros(3, device="cuda"))
            rendered_depth_map = render_pkg["depth"].squeeze().cpu().numpy()

            # 3. Tính loss và trực quan hóa
            rendered_norm, est_norm, loss_value = calculate_depth_loss(rendered_depth_map, estimated_depth)
            all_losses[view.image_name] = loss_value
            
            output_path = os.path.join(output_dir, f"comparison_{os.path.basename(view.image_name)}.png")
            visualize_and_save(rendered_norm, est_norm, output_path)

        # 4. Sau khi hoàn tất, tính loss trung bình và lưu kết quả
        avg_loss = np.mean(list(all_losses.values()))
        loss_output_path = os.path.join(output_dir, "rendered_depth_loss_results.txt")
        
        with open(loss_output_path, 'w', encoding='utf-8') as f:
            f.write(f"Kết quả so sánh Độ sâu Kết xuất (Rendered Depth) - MAE - Checkpoint: {iteration}\n")
            f.write("="*60 + "\n")
            f.write(f"Loss trung bình trên tất cả các góc nhìn: {avg_loss:.6f}\n")
            f.write("="*60 + "\n\n")
            f.write("Loss chi tiết cho từng ảnh (sắp xếp từ cao đến thấp):\n")
            sorted_losses = sorted(all_losses.items(), key=lambda item: item[1], reverse=True)
            for name, loss in sorted_losses:
                f.write(f"- {name}: {loss:.6f}\n")
        
        print(f"\nHoàn tất! Đã lưu kết quả loss vào: {loss_output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Script so sánh độ sâu kết xuất từ 3DGS và Depth Anything V2.")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int, help="Vòng lặp checkpoint cần phân tích.")
    args = get_combined_args(parser)
    
    safe_state(True)
    
    compare_rendered_depths(model_params.extract(args), pipeline_params.extract(args), args.iteration)