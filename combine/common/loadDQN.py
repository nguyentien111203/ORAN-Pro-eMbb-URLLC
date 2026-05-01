import torch
from combine.common.multiagent_DQN import MultiHeadDQN

def load_dqn_agent(model_path, state_dim, num_urllc, num_bwps, num_urllc_ue, device="cpu"):
    """
    Load DQN multi-head từ file .pth đã lưu.
    str model_path : đường dẫn file .path tới mô hình DQN
    int state_dim : số chiều của state
    int num_urllc : số slice urllc
    int num_bwps : số bwp của RU
    list(int) num_urllc_ue : list số ue mỗi slice quản lý 
    Trả về: policy_net, target_net
    """
    # Khởi tạo mạng
    policy_net = MultiHeadDQN(state_dim, num_urllc, num_bwps, num_urllc_ue).to(device)
    target_net = MultiHeadDQN(state_dim, num_urllc, num_bwps, num_urllc_ue).to(device)

    # Load trọng số
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    return policy_net, target_net