import torch

class ImageBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images):
        return_images = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                return_images.append(img)
            else:
                if torch.rand(1) < 0.5:
                    # use stored
                    idx = torch.randint(0, len(self.data), (1,)).item()
                    tmp = self.data[idx].clone()
                    self.data[idx] = img
                    return_images.append(tmp)
                else:
                    return_images.append(img)
        return torch.cat(return_images, dim=0)

