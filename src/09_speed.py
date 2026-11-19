"""Step 9 - How fast is the full pipeline on this laptop? (live / real-time check)

In live use, every new camera frame goes through ResNet ONCE; its features are kept in memory and reused
as t-k / t / t+k for later decisions. So the cost per frame = ResNet on 1 frame + BeltPulse (transformer) on 1 sample.
Camera: 120 frames per second -> budget of 8.3 ms per frame.

Usage:  .venv\\Scripts\\python src\\09_speed.py
"""
import time
import torch
import torchvision

from model import BeltPulseNet

dev = "cuda"
resnet = torchvision.models.resnet18(weights=None)
backbone = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                               resnet.layer1, resnet.layer2, resnet.layer3).eval().to(dev)
model = BeltPulseNet("beltpulse").eval().to(dev)
frame = torch.randn(1, 3, 352, 640, device=dev)
feats = torch.randn(1, 3, 256, 22, 40, device=dev)
shift = torch.tensor([[3.0, 0.0, -3.0]], device=dev)


def timeit(fn, n=200):
    with torch.no_grad():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


for half in [False, True]:
    ctx = torch.autocast("cuda", dtype=torch.float16) if half else torch.autocast("cuda", enabled=False)
    with ctx:
        a = timeit(lambda: backbone(frame))
        b = timeit(lambda: model(feats, shift))
    total = a + b
    print(f"{'float16' if half else 'float32'}: ResNet {a:5.1f} ms + BeltPulse {b:5.1f} ms = {total:5.1f} ms/frame"
          f"  -> {1000 / total:5.0f} frames/s  ({'keeps up with' if total < 8.33 else 'slower than'} 120 fps camera)")
