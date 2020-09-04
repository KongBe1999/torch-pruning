import time
import finetune
import prune
import torch
from torchstat import stat
from torchsummary import summary


if __name__ == "__main__":
    
    # device = "gpu"  # default cpu
    device = torch.device("cuda:0")
    width = 224
    height = 224

    fd = finetune.ModifiedVGG16Model()
    # fd = torch.load("/data/kong/pytorch-pruning/prune/Iteration:0.pth", map_location=lambda storage, loc: storage)
    # model.load_state_dict("/data/kong/pytorch-pruning/final-model-prunned")

    # print(fd)
    fd.eval()
    fd.to(device)
    x = torch.randn(1, 3, width, height).to(device)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(fd.to(device), (3, width, height), print_per_layer_stat=True, as_strings=True)
    # print("FLOPS:", flops)
    # print("PARAMS:", params)
    string = []
    # string.append(fd)
    string.append(f"FLOPs: {flops}\n")
    string.append(f"parameters: {params}\n")
    
    for i in range(5):
        time_time = time.time()
        features = fd(x)
        string.append("inference time: {} s \n".format(time.time() - time_time))
        

    fopen = open("result_prune", "w+")
    fopen.writelines(string)