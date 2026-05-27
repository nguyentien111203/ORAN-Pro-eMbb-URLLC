import torch
from combine.common.multiagent_DQN import MultiHeadDQN

def load_dqn_agent(model_path, state_dim, num_urllc, num_embb, num_bwps, num_urllc_ue, num_embb_ue, device="cpu"):
    """
    Load DQN multi-head từ file .pth đã lưu.
    
    """
    
    total_slices = num_urllc + num_embb
    
    # Nối 2 list số lượng UE lại với nhau 
    
    total_ues_per_slice = num_urllc_ue + num_embb_ue 
    
    # 1. Khởi tạo mạng với TỔNG số lượng slice và UE
    policy_net = MultiHeadDQN(state_dim, total_slices, num_bwps, total_ues_per_slice).to(device)
    target_net = MultiHeadDQN(state_dim, total_slices, num_bwps, total_ues_per_slice).to(device)
    
    # 2. Load trọng số
    checkpoint = torch.load(model_path, map_location=device)
    
    try:
        policy_net.load_state_dict(checkpoint)
    except RuntimeError:
        # Đề phòng trường hợp lúc train save dưới dạng dictionary
        if 'model_state_dict' in checkpoint:
            policy_net.load_state_dict(checkpoint['model_state_dict'])
        elif 'policy_net' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net'])
            
    target_net.load_state_dict(policy_net.state_dict())
    
 
    policy_net.eval()
    target_net.eval()
    
    return policy_net, target_net