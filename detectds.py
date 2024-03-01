
#coding:utf-8
#-*- coding : utf-8 -*-

# YOLOv5 馃殌 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.



Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""



import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import random
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


#colorsmap  = np.array([[0, 0, 0],
        #[204,102,153],
        #[0,255,204],
        #[102,102,255],
        #[0,0,255 ],
        #[51,102,51 ],
        #[0,255,0 ],
        #[51,153,102]], dtype=np.uint8)

colorsmap  = np.array([[117, 31, 150],
        [204,11,217],
        [197,250,37],
        [10,100,32],
        [143,227,37],
        [163,218,102],
        [147,141,120],
        [39,243,4]], dtype=np.uint8)      

        
@torch.no_grad()
def run(weights=ROOT / 'runs/train/exp5/weights/best.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    ckpts = torch.load(weights)
    # model = DetectMultiBackend(weights, device=device, dnn=dnn)
    model = ckpts['model']
    model.cuda()
    # model.half()
    stride, names = model.stride[-1].item(), model.names
    clsnum = len(names)
    colorlist = [[117,31,150],[204,11,217],[197,250,37],[10,100,32]]
    print(colorsmap)
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # # Half
    # half &= (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    # if pt:
    #     model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup


    dt, seen = [0.0, 0.0, 0.0], 0
    totaltime = 0
    for path, im, im0s, vid_cap, s in dataset:


        gain = min(im.shape[1] / im0s.shape[0], im.shape[2] / im0s.shape[1])  # gain  = old / new
        pad = (im.shape[2] - im0s.shape[1] * gain) / 2, (im.shape[1] - im0s.shape[0] * gain) / 2  # wh padding

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half()
        # im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        starttime = time.time()
        predout = model(im, augment=augment, visualize=visualize)
        endtime = time.time()
        totaltime += (endtime-starttime)
        ########################################seg#############
        segout = predout[1]
        segout = torch.nn.functional.softmax(segout, dim=1)
        segout = segout.squeeze(0)
        mask = torch.argmax(segout, dim=0)
        mask = mask.detach().cpu().numpy()
        mask = mask.astype(np.uint8)
        oldshape = mask.shape
        mask = mask[int(pad[1]):int(oldshape[0]-pad[1]), int(pad[0]):int(oldshape[1]-pad[0])]
        mask1 = cv2.resize(mask,(1024,512)) ###hanfujun
        pred_color = colorEncode(mask, colorsmap).astype(np.uint8)
        t3 = time_sync()
        dt[1] += t3 - t2
        pred = non_max_suppression(predout[0][0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            record_xyxy = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                #record_xyxy = [] ###hanfujun
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #with open(txt_path + '.txt', 'a') as f:  ###hanfujun
                            #f.write(('%g ' * len(line)).rstrip() % line + '\n') ###hanfujun


                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            for i in range(len(xyxy)): ###hanfujun
                                xyxy[i] = int(xyxy[i].cpu().numpy()) ###hanfujun
                            record_xyxy.append(xyxy) ###hanfujun
            
            """
            panduan
            """
            record_xyxy = np.array(record_xyxy) ###hanfujun
            imgsum = []
            imgone = []
            for i in range(len(record_xyxy)):
                x1 = record_xyxy[i][0]
                y1 = record_xyxy[i][1]
                x2 = record_xyxy[i][2]
                y2 = record_xyxy[i][3]
                imgsum.append(mask1[y1:y2, x1:x2])
                imgone.append(np.ones((y2-y1, x2-x1), dtype=np.uint8))
            yes_or_no = []
            for i in range(len(record_xyxy)):
                a = imgsum[i]
                b = imgone[i]
                new = np.bitwise_and(a, b)
                if np.sum(new) > 20:
                #if np.sum(new) > 150:
                    yes_or_no.append('Y')
                else:
                    yes_or_no.append('N')
                    
            """
            xieru
            """
            idx = 0
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                     with open(txt_path + '.txt', 'a') as f:  ###hanfujun
                         f.write(('%g ' * len(line)).rstrip() % line + ' ' + yes_or_no[idx] + '\n') ###hanfujun
                if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        label_split = label.split()
                        if label_split[1] == 'Vehicle,Bicycle':
                           label = yes_or_no[idx] + ',' + 'T,B'
                        elif label_split[1] == 'Vehicle,Motorcycle':
                           label = yes_or_no[idx] + ',' + 'T,M'
                        if label_split[0] == 'Pedestrian':
                           label = yes_or_no[idx] + ',' + 'P,P'
                        elif label_split[0] == 'Person,Rider':
                           label = yes_or_no[idx] + ',' + 'P,R'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                idx += 1
            print(yes_or_no) 
            
            

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')



            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                   # cv2.imwrite(save_path, im0)

                    maskcolor = cv2.resize(pred_color,(im0.shape[1],im0.shape[0]))
                    im_vis = cv2.addWeighted(im0,0.5,maskcolor,0.5,1);
                    cv2.imwrite("{}_yolods.png".format(save_path[:-4]),im_vis)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    print("total image : ",seen, ";  total time : ",totaltime,"s; average inference time pre image : ",totaltime/seen)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'run/exp/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/imges/ ', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='6', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False,action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=True, action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize',default=False, action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect/', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
