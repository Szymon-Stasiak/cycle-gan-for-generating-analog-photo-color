import random
import torch

class ImageBuffer:
    def __init__(self, max_size=50, device=None):
        self.max_size = max_size
        self.data = []
        self.device = device

    def push_and_pop(self, images):
        if self.device is None:
            self.device = images.device
        returned = []
        for img in images.detach():
            img = img.unsqueeze(0)
            img_cpu = img.detach().clone().to(self.device)
            if len(self.data) < self.max_size:
                self.data.append(img_cpu)
                returned.append(img_cpu)
            else:
                if random.uniform(0,1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = img_cpu
                    returned.append(tmp)
                else:
                    returned.append(img_cpu)
        out = torch.cat(returned, dim=0).to(images.device)
        return out
