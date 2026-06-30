import torch
from combine.common.common import GaussianPolicy, MLP

def load_sac_components(sac_model_path, state_dim, num_RU, num_slices,
                        action_scale, action_bias, device="cpu"):
    """
    Load các thành phần của SAC từ file .pth đã lưu.
    Trả về: actor, critic_1, critic_2, alpha
    """
    checkpoint = torch.load(sac_model_path, map_location=device)
    # Khởi tạo actor đúng kiến trúc
    actor = GaussianPolicy(state_dim, num_RU * num_slices,
                           action_scale=action_scale,
                           action_bias=action_bias).to(device)
    actor.load_state_dict(checkpoint['actor'])

    # Khởi tạo critics
    input_dim = state_dim + num_RU * num_slices
    critic_1 = MLP(input_dim, 1).to(device)
    critic_2 = MLP(input_dim, 1).to(device)
    critic_1.load_state_dict(checkpoint['critic_1'])
    critic_2.load_state_dict(checkpoint['critic_2'])

    # Load alpha nếu có
    alpha = checkpoint.get('alpha', 0.2)  # fallback nếu chưa lưu

    return actor, critic_1, critic_2, alpha