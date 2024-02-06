from models.KronLinear import KronLinear, KronMlp, Attention, Block
import torch 


if __name__ == "__main__":
    kronmlp = KronMlp(4, 4, 4)
    kronlinear = KronLinear(4, 4)
    x = torch.randn(1, 1, 4)
    print(kronlinear(x).shape)    
    kronattn = Attention(dim=4, num_heads=2)
    print(kronattn(x).shape)
    block = Block(4, 2)
    print(block(x).shape)
