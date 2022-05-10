import torch
from torch.autograd import Variable
import cv2
import imageio


def detect(frame, net, transform):

    height, width, channel = frame.shape
    frame_transformed = transform(frame)[0]
    x = torch.from_numpy(frame_transformed).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])


    return frame


if __name__ == '__main__':
    detect()
