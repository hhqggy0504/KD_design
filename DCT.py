import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from torchvision import transforms
from PIL import Image


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

feature_model=Down_wt(3,3)

# if __name__ == '__main__':
#     block = Down_wt(3, 3)  # Adjusted for image input with 3 channels (RGB)
#
#     # Load input image
#     input_image_path = './FUSAR1/FUSAR/CargoShip/Ship_C01S07N0001.tiff'  # Replace with your image path
#     input_tensor = image_to_tensor(input_image_path)
#
#     # Get the output tensor from the model
#     output_tensor = block(input_tensor)
#
#     # Convert the output tensor back to an image
#     output_image = tensor_to_image(output_tensor)
#
#     # Save the output image
#     output_image.save('output_image.jpg')
#     print("Output image saved successfully.")
