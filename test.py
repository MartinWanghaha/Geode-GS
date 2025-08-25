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

# --- 示例用法 (使用更能说明问题的模拟数据) ---
if __name__ == "__main__":
    # 导入可视化库
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not installed. Cannot visualize results.")
        exit()

    # --- 1. 创建一个更清晰的模拟场景 ---
    H, W = 256, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # a. 创建 Prior Normal (引导图): 一个平滑背景上有一个平滑但倾斜的方块
    #    特点：内部平滑，边界锐利。模拟了神经网络的平滑输出。
    
    # 背景法线 (大部分朝向Z轴)
    bg_normal = torch.tensor([0.0, 0.0, 1.0], device=device).view(3, 1, 1)
    prior_normal_tensor = bg_normal.repeat(1, H, W)

    # 前景方块法线 (向左上方倾斜)
    obj_normal = torch.tensor([-0.6, 0.5, 0.8], device=device).view(3, 1, 1)
    obj_normal = F.normalize(obj_normal, p=2, dim=0) # 确保是单位向量
    
    # 定义方块区域
    x1, y1 = H // 4, W // 4
    x2, y2 = H - H // 4, W - W // 4
    prior_normal_tensor[:, y1:y2, x1:x2] = obj_normal

    print(f"Prior Normal (Guidance) created with sharp edges. Shape: {prior_normal_tensor.shape}")

    # b. 创建 Depth Normal (输入图): 在Prior Normal的结构基础上添加大量噪声
    #    特点：保留了底层结构，但充满了高频噪声。模拟了从原始深度图计算的粗糙法线。
    noise = torch.randn(3, H, W, device=device) * 0.4 # 较强的噪声
    depth_normal_tensor = prior_normal_tensor + noise
    depth_normal_tensor = F.normalize(depth_normal_tensor, p=2, dim=0)

    print(f"Depth Normal (Input) created with added noise. Shape: {depth_normal_tensor.shape}")

    # --- 2. 执行融合和对比实验 ---
    
    # a. 使用我们的引导融合函数
    fused_normal_tensor = guided_normal_fusion_corrected(
        depth_normal_tensor,
        prior_normal_tensor,
        kernel_size=15, # 使用较大的核来强调平滑效果
        eps=1e-3       # 降低eps可以增强边缘保留效果
    )
    print("Guided fusion complete.")

    # b. 对比：使用标准高斯模糊
    # Kornia期望(N,C,H,W)，所以先添加batch维度
    gaussian_blurred_normal = KF.gaussian_blur2d(
        depth_normal_tensor.unsqueeze(0),
        kernel_size=(15, 15),
        sigma=(5.0, 5.0)
    ).squeeze(0) # 移除batch维度
    gaussian_blurred_normal = F.normalize(gaussian_blurred_normal, p=2, dim=0)
    print("Gaussian blur comparison complete.")


    # --- 3. 可视化结果进行对比 ---
    
    # 将 [-1, 1] 的法线值映射到 [0, 1] 的RGB值用于显示
    def visualize_normal(normal_tensor):
        normal_np = normal_tensor.cpu().numpy()
        normal_np = np.transpose(normal_np, (1, 2, 0))
        return (normal_np + 1) / 2

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(visualize_normal(prior_normal_tensor))
    plt.title('1. Prior Normal (Guidance)\n(Clean Structure)')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(visualize_normal(depth_normal_tensor))
    plt.title('2. Depth Normal (Input)\n(Structure + Noise)')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(visualize_normal(fused_normal_tensor))
    plt.title('3. Fused Normal (Our Result)\n(Structure Preserved, Noise Removed)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(visualize_normal(gaussian_blurred_normal))
    plt.title('4. Gaussian Blur (Comparison)\n(Edges Blurred, Structure Lost)')
    plt.axis('off')

    # 调整子图布局，防止标题重叠
    plt.tight_layout()

    # --- 保存图像 ---
    # 定义保存路径和文件名。文件格式由扩展名决定 (例如 .png, .jpg, .pdf, .svg)
    output_filename = 'fang_normal_fusion_comparison.png'

    # 调用 savefig 函数保存图像
    # dpi (dots per inch) 控制图像的分辨率，值越高图像越清晰，文件也越大。300是常见的出版质量。
    # bbox_inches='tight' 会自动裁剪掉图像周围多余的白边，非常有用。
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"可视化结果已保存到: {output_filename}")

    # --- 显示图像 ---
    # 仍然可以调用 plt.show() 在屏幕上显示图像
    plt.show()