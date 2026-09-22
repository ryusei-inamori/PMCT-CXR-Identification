import torch
from pmct_cxr_id.model import CNN

def test_embedding_shapes():
    m=CNN(n_classes=10,hidden_dim=1280).eval()
    x=torch.zeros(2,1,320,320)
    with torch.no_grad():
        assert m.encode(x,"paper1280").shape == (2,1280)
        assert m.encode(x,"preadacos1536").shape == (2,1536)
        assert m.encode(x,"backbone1536").shape == (2,1536)
