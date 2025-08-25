import torch
import torch.nn.functional as F
import kornia.filters as KF
import kornia.geometry.transform as KGT # 用于可能需要的维度操作，虽然这里不直接用

def guided_normal_fusion_corrected(
    depth_normal: torch.Tensor,
    prior_normal: torch.Tensor,
    kernel_size: int = 11, # kernel_size for guided_blur typically an odd integer, e.g., 5x5 -> 5
    eps: float = 1e-6,
    border_type: str = 'reflect' # new parameter for guided_blur
) -> torch.Tensor:
    """
    使用 kornia.filters.guided_blur 融合 depth_normal 和 prior_normal。

    Args:
        depth_normal (torch.Tensor): 从深度图计算得到的法线，形状为 (3, H, W)，值域应为 [-1, 1]。
                                      通常表示为 (N, C, H, W) 或 (C, H, W)。
        prior_normal (torch.Tensor): 深度学习模型预测的法线，形状为 (3, H, W)，值域应为 [-1, 1]。
                                      通常表示为 (N, C, H, W) 或 (C, H, W)。
        kernel_size (int): 引导模糊核的大小 (例如，5 表示 5x5 核)。必须是奇数。
                          越大越平滑，但可能丢失细节。
        eps (float): 引导模糊的正则化参数，用于防止过拟合，避免除以零。
        border_type (str): 图像边界填充模式。可以是 'constant', 'reflect', 'replicate', 'circular'。

    Returns:
        torch.Tensor: 融合后的法线，形状为 (3, H, W)，值域为 [-1, 1]。
    """
    # 确保输入张量在正确的设备上 (CPU 或 GPU)
    device = depth_normal.device
    
    # 确保输入是浮点型
    if not depth_normal.is_floating_point():
        depth_normal = depth_normal.float()
    if not prior_normal.is_floating_point():
        prior_normal = prior_normal.float()

    # Kornia 的引导模糊期望 (N, C, H, W) 格式，如果输入是 (C, H, W) 则添加批次维度
    original_dim = depth_normal.dim()
    if original_dim == 3:
        depth_normal = depth_normal.unsqueeze(0)  # 添加批次维度
        prior_normal = prior_normal.unsqueeze(0)  # 添加批次维度

    # 检查张量形状是否匹配
    if depth_normal.shape != prior_normal.shape:
        raise ValueError(
            f"Input tensors must have the same shape. Got depth_normal: {depth_normal.shape} "
            f"and prior_normal: {prior_normal.shape}"
        )
    
    # 确保通道数为3 (RGB或XYZ)
    if depth_normal.shape[1] != 3:
        raise ValueError(f"Input tensors must have 3 channels (for X, Y, Z normal components). "
                         f"Got {depth_normal.shape[1]} channels.")
    
    # 确保 kernel_size 是奇数
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd integer. Got {kernel_size}.")

    # 归一化法线向量，确保它们是单位向量
    depth_normal_norm = F.normalize(depth_normal, p=2, dim=1)
    prior_normal_norm = F.normalize(prior_normal, p=2, dim=1)

    # 应用引导模糊
    # input: depth_normal_norm (要被模糊的图像)
    # guidance: prior_normal_norm (提供结构信息的引导图像)
    # kernel_size: 整数，表示方形核的边长，例如 5 -> 5x5 核
    # eps: 正则化参数
    fused_normal = KF.guided_blur(
        input=depth_normal_norm,
        guidance=prior_normal_norm,
        kernel_size=(kernel_size, kernel_size), # kornia expects a tuple for kernel_size
        eps=eps,
        border_type=border_type
    )

    # 再次归一化融合后的法线，因为滤波过程可能会使其不再是单位向量
    fused_normal = F.normalize(fused_normal, p=2, dim=1)

    # 如果输入是 (C, H, W)，则移除批次维度
    if original_dim == 3:
        fused_normal = fused_normal.squeeze(0)

    return fused_normal


# --- 示例用法: 从本地文件加载图像并执行融合 ---
if __name__ == "__main__":
    # 导入必要的库
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        import torchvision.transforms as T
    except ImportError as e:
        print(f"必要的库未安装，请运行 'pip install Pillow torchvision matplotlib numpy'。错误: {e}")
        exit()

    # --- 1. 定义加载和预处理函数 ---
    def load_normal_map(image_path: str, device: torch.device) -> torch.Tensor:
        """
        从文件加载法线图，并进行预处理。
        - 将图像像素值 [0, 255] 转换为张量
        - 将范围从 [0, 1] 映射到 [-1, 1]
        - 归一化为单位向量
        """
        try:
            # 使用Pillow加载图像，并确保为RGB格式
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"错误：找不到文件 '{image_path}'。请检查路径是否正确。")
            return None

        # 定义转换流程：PIL Image -> PyTorch Tensor
        transform = T.ToTensor() # 将(H,W,C)的PIL图像 [0,255] 转换为 (C,H,W)的Tensor [0,1]
        
        # 应用转换
        normal_tensor = transform(image)

        # 将范围从 [0, 1] 映射到 [-1, 1]
        normal_tensor = normal_tensor * 2.0 - 1.0

        # 归一化法向量，确保它们是单位向量（这是一个好习惯）
        normal_tensor = F.normalize(normal_tensor, p=2, dim=0)

        # 将张量移动到指定设备 (CPU 或 GPU)
        return normal_tensor.to(device)

    # --- 2. 加载您的图像 ---
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # !! 修改为您自己的文件路径 !!
    prior_normal_path = './02250_DSC08057_prior_normal_show.jpg' # 引导图 (例如，来自AI模型的平滑法线)
    depth_normal_path = './02250_DSC08057_depth_normal_show.jpg' # 输入图 (例如，从深度图计算的带噪声的法线)

    prior_normal_tensor = load_normal_map(prior_normal_path, device)
    depth_normal_tensor = load_normal_map(depth_normal_path, device)

    # 检查图像是否成功加载
    if prior_normal_tensor is None or depth_normal_tensor is None:
        print("图像加载失败，程序终止。")
        exit()
        
    print(f"成功加载 Prior Normal, 形状: {prior_normal_tensor.shape}")
    print(f"成功加载 Depth Normal, 形状: {depth_normal_tensor.shape}")

    # --- 3. 执行融合和对比实验 ---
    
    # a. 使用引导融合函数
    fused_normal_tensor = guided_normal_fusion_corrected(
        depth_normal_tensor,
        prior_normal_tensor,
        kernel_size=15,
        eps=1e-3
    )
    print("引导融合处理完成。")

    # b. 对比：使用标准高斯模糊
    gaussian_blurred_normal = KF.gaussian_blur2d(
        depth_normal_tensor.unsqueeze(0),
        kernel_size=(15, 15),
        sigma=(5.0, 5.0)
    ).squeeze(0)
    gaussian_blurred_normal = F.normalize(gaussian_blurred_normal, p=2, dim=0)
    print("高斯模糊对比处理完成。")


    # --- 4. 可视化结果 ---
    def visualize_normal(normal_tensor):
        # 确保张量在CPU上并转换为numpy数组
        normal_np = normal_tensor.cpu().numpy()
        # 从 (C, H, W) 转换到 (H, W, C) 以便matplotlib显示
        normal_np = np.transpose(normal_np, (1, 2, 0))
        # 将 [-1, 1] 范围映射回 [0, 1] 用于可视化
        return (normal_np + 1) / 2

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(visualize_normal(prior_normal_tensor))
    plt.title('1. Prior Normal (Guidance)')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(visualize_normal(depth_normal_tensor))
    plt.title('2. Depth Normal (Input)')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(visualize_normal(fused_normal_tensor))
    plt.title('3. Fused Normal (Result)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(visualize_normal(gaussian_blurred_normal))
    plt.title('4. Gaussian Blur (Comparison)')
    plt.axis('off')

    # 调整子图布局，防止标题重叠
    plt.tight_layout()

    # --- 保存图像 ---
    # 定义保存路径和文件名。文件格式由扩展名决定 (例如 .png, .jpg, .pdf, .svg)
    output_filename = 'normal_fusion_comparison.png'

    # 调用 savefig 函数保存图像
    # dpi (dots per inch) 控制图像的分辨率，值越高图像越清晰，文件也越大。300是常见的出版质量。
    # bbox_inches='tight' 会自动裁剪掉图像周围多余的白边，非常有用。
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"可视化结果已保存到: {output_filename}")

    # --- 显示图像 ---
    # 仍然可以调用 plt.show() 在屏幕上显示图像
    plt.show()