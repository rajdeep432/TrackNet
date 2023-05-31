from argparse import ArgumentParser
from TrackNet import TrackNet
import torchvision
import torch
import cv2

def get_ball_position(img, opt, original_img_=None):
    ret, thresh = cv2.threshold(img, opt.brightness_thresh, 1, 0)
    thresh = cv2.convertScaleAbs(thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # Find the biggest area of the contour
        c = max(contours, key=cv2.contourArea)

        if original_img_ is not None:
            # Draw the contours
            cv2.drawContours(original_img_, [c], -1, 255, 3)

        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('video', type=str, default='video.mp4', help='Path to video.')
    parser.add_argument('--save_path', type=str, default='predicted.mp4', help='Path to result video.')
    parser.add_argument('--weights', type=str, default='weights', help='Path to trained model weights.')
    parser.add_argument('--sequence_length', type=int, default=3, help='Length of the image sequence used as X.')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024], help='Size of the images used for training (y, x).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps).')
    parser.add_argument('--one_output_frame', action='store_true', help='Demand only one output frame instead of three.')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images instead of RGB.')
    parser.add_argument('--visualize', action='store_true', help='Display the predictions in real time.')
    parser.add_argument('--waitBetweenFrames', type=int, default=100, help='Wait time in milliseconds between showing frames predicted in one forward pass.')
    parser.add_argument('--brightness_thresh', type=int, default=0.7, help='Result heatmap pixel brightness threshold')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    opt.dropout = 0
    device = torch.device(opt.device)
    model.load('example_datasets/video_dataset/csvs')
    model = TrackNet(opt).to(device)
    #model = TrackNet()
    model.load_state_dict(opt.weights, device=opt.device)
    model.eval()
    


    cap = cv2.VideoCapture(opt.video)
    videoEndReached = False

    while cap.isOpened():
        # TODO: add one_output_frame support

        frames = []
        for _ in range(opt.sequence_length):
            ret, frame = cap.read()
            if not ret:
                videoEndReached = True
                break
            frames.append(frame)

        if videoEndReached:
            break

        frames_torch = []
        for frame in frames:
            frame_torch = torch.tensor(frame).permute(2, 0, 1).float().to(device) / 255
            frame_torch = torchvision.transforms.functional.resize(frame_torch, opt.image_size)
            frames_torch.append(frame_torch)

        frames_torch = torch.cat(frames_torch, dim=0).unsqueeze(0)

        pred = model(frames_torch)
        pred = pred[0, :, :, :].detach().cpu().numpy()

        for i in range(opt.sequence_length):
            pred_frame = pred[i, :, :]
            pred_frame = cv2.resize(pred_frame, (frames[i].shape[1], frames[i].shape[0]), interpolation=cv2.INTER_AREA)
            get_ball_position(pred_frame, opt, original_img_=frames[i])
            cv2.imshow('prediction', pred_frame)
            cv2.imshow('original', frames[i])
            cv2.waitKey(opt.waitBetweenFrames)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows
