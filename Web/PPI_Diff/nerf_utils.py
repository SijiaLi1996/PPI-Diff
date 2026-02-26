import torch
import torch.nn.functional as F
import math

def nerf_extend(a, b, c, l_cd, theta, chi):
    """
    NeRF (Natural Extension Reference Frame) 核心算法
    根据前三个原子 a, b, c，利用键长 l_cd，键角 theta，二面角 chi 计算第四个原子 d
    """
    # 1. 构建局部坐标系 (bc 为 x轴)
    bc = c - b
    bc_unit = F.normalize(bc, dim=-1)
    
    n = torch.cross(b - a, bc) # 法向量
    n_unit = F.normalize(n, dim=-1)
    
    m = torch.cross(bc_unit, n_unit) # z轴 (或 y轴，取决于定义)
    
    # 2. 在局部坐标系下计算 d 的位置 (d2)
    # d2 = [ -l * cos(theta), l * sin(theta) * cos(chi), l * sin(theta) * sin(chi) ]
    # 注意：这里的 theta 通常定义为补角，或者直接是键角，需根据习惯调整
    # 这里假设 theta 是键角 (例如 N-Ca-C ~ 111度)
    # 这里的 chi 是二面角
    
    # 为了简化计算，我们在标准 NeRF 定义下：
    # D = C + l * ( -cos(pi-theta)*bc_unit + sin(pi-theta)*cos(chi)*n_unit + sin(pi-theta)*sin(chi)*m )
    
    # 预计算三角函数
    # theta 是键角，NeRF 旋转通常用补角 (180 - theta)
    # 但在代码实现中，直接用 cos/sin 组合即可
    
    d2_x = -l_cd * torch.cos(torch.tensor(math.pi) - theta)
    d2_y = l_cd * torch.sin(torch.tensor(math.pi) - theta) * torch.cos(chi)
    d2_z = l_cd * torch.sin(torch.tensor(math.pi) - theta) * torch.sin(chi)
    
    # 3. 变换回全局坐标系
    # d = c + x*bc_unit + y*n_unit + z*m
    d = c + d2_x * bc_unit + d2_y * n_unit + d2_z * m
    return d

def reconstruct_backbone_from_angles(angles):
    """
    将模型预测的 12D 角度特征转换为 N, CA, C 骨架坐标。
    
    Args:
        angles: [SeqLen, 12] tensor. 
                假设顺序为: 
                0-1: phi (sin, cos)
                2-3: psi (sin, cos)
                4-5: omega (sin, cos)
                6-11: 预留给键角 (tau, theta1, theta2) 的 sin, cos
                
    Returns:
        coords: [SeqLen * 3, 3] tensor (N, CA, C, N, CA, C ...)
    """
    L = angles.shape[0]
    device = angles.device
    
    # --- 1. 解码角度 (sin/cos -> radians) ---
    # 只要前 3 个二面角 (phi, psi, omega)
    # 和 3 个键角 (N-CA-C, CA-C-N, C-N-CA)
    
    # phi: C(i-1) - N(i) - CA(i) - C(i)
    # psi: N(i) - CA(i) - C(i) - N(i+1)
    # omega: CA(i) - C(i) - N(i+1) - CA(i+1)
    
    # 从 12D 中提取 6 个角度
    # [phi, psi, omega, tau(N-Ca-C), theta1(Ca-C-N), theta2(C-N-Ca)]
    decoded_angles = []
    for i in range(6):
        s = angles[:, 2*i]
        c = angles[:, 2*i+1]
        decoded_angles.append(torch.atan2(s, c))
    
    phi, psi, omega, tau, theta1, theta2 = decoded_angles
    
    # --- 2. 定义标准键长 (Angstrom) ---
    L_N_CA = 1.458
    L_CA_C = 1.525
    L_C_N  = 1.329
    
    # --- 3. 初始化 ---
    # 我们需要存储 N, CA, C 三个原子
    coords = torch.zeros((L * 3, 3), device=device)
    
    # 放置第一个残基的前三个原子 (作为参考系)
    # 假设第一个 N 在原点
    coords[0] = torch.tensor([0., 0., 0.], device=device) # N_0
    coords[1] = torch.tensor([L_N_CA, 0., 0.], device=device) # CA_0
    
    # 第三个原子 C_0 由第一个键角 tau_0 决定
    # 位于 xy 平面
    # C = CA + L_CA_C * [cos(180-tau), sin(180-tau), 0]
    t0 = tau[0]
    coords[2] = coords[1] + torch.tensor([
        -L_CA_C * torch.cos(math.pi - t0),
        L_CA_C * torch.sin(math.pi - t0),
        0.
    ], device=device) # C_0
    
    # --- 4. 递归生成后续原子 ---
    # 顺序: N(i), CA(i), C(i)
    # 已知: N(i-1), CA(i-1), C(i-1) -> 生成 N(i) (利用 psi(i-1), theta1(i-1))
    # 已知: CA(i-1), C(i-1), N(i)   -> 生成 CA(i) (利用 omega(i-1), theta2(i-1))
    # 已知: C(i-1), N(i), CA(i)     -> 生成 C(i)  (利用 phi(i), tau(i))
    
    for i in range(1, L):
        # 索引
        idx_prev_n  = (i-1)*3
        idx_prev_ca = (i-1)*3 + 1
        idx_prev_c  = (i-1)*3 + 2
        
        idx_curr_n  = i*3
        idx_curr_ca = i*3 + 1
        idx_curr_c  = i*3 + 2
        
        # 获取前三个原子坐标
        prev_n  = coords[idx_prev_n]
        prev_ca = coords[idx_prev_ca]
        prev_c  = coords[idx_prev_c]
        
        # 1. 生成当前 N (连接 C(i-1) - N(i))
        # 键长: L_C_N
        # 键角: theta1 (Ca-C-N) -> 对应 angles index 4
        # 二面角: psi (N-Ca-C-N) -> 对应 angles index 1
        # 注意：这里的 psi 是 i-1 的 psi
        curr_n = nerf_extend(prev_n, prev_ca, prev_c, L_C_N, theta1[i-1], psi[i-1])
        coords[idx_curr_n] = curr_n
        
        # 2. 生成当前 CA (连接 N(i) - CA(i))
        # 键长: L_N_CA
        # 键角: theta2 (C-N-Ca) -> 对应 angles index 5
        # 二面角: omega (Ca-C-N-Ca) -> 对应 angles index 2
        # 注意：这里的 omega 是 i-1 的 omega
        curr_ca = nerf_extend(prev_ca, prev_c, curr_n, L_N_CA, theta2[i-1], omega[i-1])
        coords[idx_curr_ca] = curr_ca
        
        # 3. 生成当前 C (连接 CA(i) - C(i))
        # 键长: L_CA_C
        # 键角: tau (N-Ca-C) -> 对应 angles index 3
        # 二面角: phi (C-N-Ca-C) -> 对应 angles index 0
        # 注意：这里的 phi 是当前的 phi(i)
        curr_c = nerf_extend(prev_c, curr_n, curr_ca, L_CA_C, tau[i], phi[i])
        coords[idx_curr_c] = curr_c
        
    return coords