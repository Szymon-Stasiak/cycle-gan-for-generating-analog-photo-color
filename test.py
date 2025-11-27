import cv2
import torch
import numpy as np
from model.lut import LUT3D, train_lut, image_resizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def apply_lut_cv2(input_path, output_path):
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Cannot load {input_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_tensor = (
        torch.from_numpy(rgb).float().div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out = lut(rgb_tensor)

    out_np = (
        out.squeeze(0)
        .permute(1, 2, 0)
        .cpu().numpy()
    )
    out_bgr = cv2.cvtColor((out_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imshow("image", np.hstack([bgr,out_bgr]))
    cv2.waitKey(0)
    cv2.imwrite(output_path, out_bgr)
    print("Saved:", output_path)

if __name__ == "__main__":
    lut = LUT3D(dim=33, mode='txtLoad' ,path="./TrainedLUT33.txt").to(device)
    #lut.load_state_dict(torch.load("lut_cinema.pth"))
    lut.eval()
    #image_resizer(["data/input","data/cinema"], 256)   #use this if training with different datasets otherwise will take too much time
    #lut_cinema = train_lut("data/input","data/cinema", epochs=30, batch=32, workers=8)
    '''
    torch.save(lut_cinema.state_dict(),"lut_cinema.pth")    #stores the hole LUT, uncomment the line bellow LUT3D call to load it

                            OR

    lut_cinema = lut.LUT.detach().cpu().numpy()             #stores values in a txt just like the Identity, change path in LUT3D call to use it
    with open("TrainedLUT33, "w") as f:
        for i in range(lut.shape[1]):
            for j in range(lut.shape[2]):
                for k in range(lut.shape[3]):
                    r = lut[0,i,j,k]
                    g = lut[1,i,j,k]
                    b = lut[2,i,j,k]
                    f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
    '''
    apply_lut_cv2("data/test.png", "data/test1.png")
    apply_lut_cv2("data/build.jpg", "data/build1.jpg")
    #apply_lut_cv2("data/tree.jpeg", "data/tree1.jpeg")
    #apply_lut_cv2("data/smth.jpg", "data/smth1.jpg")