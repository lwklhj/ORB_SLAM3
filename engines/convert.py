from superpoint import SuperPoint
import torch
import os

dir = os.path.dirname(__file__)

# Load model
model = SuperPoint(SuperPoint.default_config)

model.load_state_dict(torch.load(os.path.join(dir, "weights/superpoint_v1.pth")))
model.eval()

sm = torch.jit.script(model)
sm.save(os.path.join(dir, "superpoint_v1.pt"))