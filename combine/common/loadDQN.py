import torch
from combine.common.multiagent_DQN import MultiHeadDQN

def load_dqn_agent(model_path, state_dim, num_PRB, num_slices, device="cpu"):
    """
    Load DQN multi-head từ file .pth đã lưu.
    Trả về: policy_net, target_net
    """
    # Khởi tạo mạng
    policy_net = MultiHeadDQN(state_dim, num_PRB, num_slices).to(device)
    target_net = MultiHeadDQN(state_dim, num_PRB, num_slices).to(device)

    # Load trọng số
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    return policy_net, target_net