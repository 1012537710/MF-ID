
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
    # print("stride:",stride,imgsz)
    # stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    clsnum = len(names)
    colorlist = [[117,31,150],[204,11,217],[197,250,37],[10,100,32]]
    #colorlist = [[255,255,255],[0,0,255],[0,255,255],[0,0,255]]
    #for i in range(clsnum):
        #colorlist.append([random.randint(0,255),random.randint(0,255),random.randint(0,255)])
    #colorsmap  = np.array(colorlist, dtype=np.uint8)
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


    ADA = [] # hanfujun 绗笁绡囨枃绔燗DA鐨勫瓨锟?涓轰簡鍚庨潰鐨勬帓锟?
    dt, seen = [0.0, 0.0, 0.0], 0
    totaltime = 0
    for path, im, im0s, vid_cap, s in dataset:

        # hanfujun 绗笁绡囨枃绔燗DA 涓轰簡灏嗘枃浠跺す涓帹鐞嗙収鐗囩殑ID瀛樿捣锟?
        number = os.path.splitext(os.path.basename(s))[0]

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
        # print(predout.shape)
        #闊╅Ε楠忔柊鍔犵殑
        ActiveLearning_pred = predout[0][0] # 杩欐槸涓€涓猼ensor
        #print(ActiveLearning_pred.shape)
        ActiveLearning_pred_cpu = ActiveLearning_pred.cpu().detach().numpy()
        # 鎻愬彇姣忎釜鏍锋湰鐨勬鐜囧垎甯冮儴锟?
        # # 杩欓噷 鎴戜滑鍏堜繚璇佸鐜板叾浠栨椂鍊欑殑锛屽洜涓哄彧闇€瑕佹鐜囷紝鎵€浠ユ垜浠殑棰勬祴鍊兼槸ActiveLearning_pred_cpu[0, :, 5:9]
        # probabilities = ActiveLearning_pred_cpu[0, :, 5:9]

        probabilities = ActiveLearning_pred_cpu[0, :, 0:9]
        #print('fsadf', probabilities.shape)

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # def calculate_total_similarity(probabilities):
        #     n_samples = probabilities.shape[0]
        #     total_similarity = 0
        #     count = 0

        #     for i in range(n_samples):
        #         for j in range(i+1, n_samples):
        #             prob_i = probabilities[i, -4:]  # 鑾峰彇绗琲涓牱鏈殑绫诲埆姒傜巼
        #             prob_j = probabilities[j, -4:]  # 鑾峰彇绗琷涓牱鏈殑绫诲埆姒傜巼
        #             similarity = cosine_similarity([prob_i], [prob_j])[0][0]
        #             total_similarity += similarity
        #             count += 1

        #     return total_similarity / count if count > 0 else 0

        # # 绀轰緥鐢ㄦ硶
        # total_similarity = calculate_total_similarity(probabilities)

        # print("鎬讳綋鐩镐技锟?", total_similarity)



        #########涓嬮潰鏄嚜宸辩殑瀹為獙1Similarity###############
        # def calculate_total_similarity_vectorized(probabilities):
        #     # 鍙€冭檻绫诲埆姒傜巼閮ㄥ垎
        #     class_probabilities = probabilities[:, -4:]
        #     # 璁＄畻鐩镐技搴︾煩锟?
        #     similarities = cosine_similarity(class_probabilities)
        #     # 蹇界暐瀵硅绾垮厓绱犲苟鍙栦笂涓夎鐭╅樀鐨勫钩鍧囷拷?
        #     upper_triangle_indices = np.triu_indices_from(similarities, k=1)
        #     average_similarity = np.mean(similarities[upper_triangle_indices])
        #     return average_similarity
        # # 绀轰緥鐢ㄦ硶
        # total_similarity = calculate_total_similarity_vectorized(probabilities)
        # # print("鎬讳綋鐩镐技锟?", total_similarity)
        # CONBINED = [total_similarity, number]
        # ADA.append(CONBINED)

        # # #########谱聚类聚类代码###############
        # import numpy as np
        # from sklearn.cluster import SpectralClustering
        # from sklearn.decomposition import PCA
        # import matplotlib.pyplot as plt
        # from scipy.stats import entropy
        # from sklearn.manifold import TSNE

        # features = probabilities[:, 4:]
        # # 使用 SpectralClustering 进行谱聚类
        # n_clusters = 4  # 聚类的数量
        # spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10,
        #                               random_state=0)
        # labels = spectral.fit_predict(features)
        #
        # # 使用t-SNE进行降维
        # tsne = TSNE(n_components=2, random_state=42)
        # reduced_features_tsne = tsne.fit_transform(features)

        # # 可视化t-SNE降维后的聚类结果
        # plt.figure(figsize=(10, 8))
        # for i in range(n_clusters):
        #     cluster_data = reduced_features_tsne[labels == i]
        #     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
        # plt.legend()
        # plt.title('t-SNE Clustering Results')
        # plt.xlabel('t-SNE Feature 1')
        # plt.ylabel('t-SNE Feature 2')
        # plt.savefig('/root/YePeng/yolov5ds/111111/聚类结果可视化.png')
        # plt.close()
        #
        # # 计算每个聚类的不确定性
        # uncertainties = []
        # for i in range(n_clusters):
        #     cluster_indices = np.where(labels == i)[0]
        #     cluster_confidences = probabilities[cluster_indices, 4]
        #     cluster_class_probabilities = probabilities[cluster_indices, 5:]
        #     confidence_uncertainty = np.std(cluster_confidences)
        #     entropy_uncertainty = np.mean([entropy(sample) for sample in cluster_class_probabilities])
        #     cluster_uncertainty = confidence_uncertainty + entropy_uncertainty
        #     uncertainties.append(cluster_uncertainty)
        # 整体图像不确定性评分
        # image_uncertainty_score = np.mean(uncertainties)
        # print("Image Uncertainty Score:", image_uncertainty_score)

        # # 置信度分布图保存
        # fig, axes = plt.subplots(1, n_clusters, figsize=(20, 5), sharey=True)
        # for i in range(n_clusters):
        #     cluster_confidences = probabilities[labels == i, 4]
        #     axes[i].hist(cluster_confidences, bins=20, alpha=0.7)
        #     axes[i].set_title(f'Cluster {i} Confidence Distribution')
        #     axes[i].set_xlabel('Confidence')
        #     axes[i].set_ylabel('Frequency')
        # plt.tight_layout()
        # plt.savefig('/root/YePeng/yolov5ds/111111/置信度分布图.png')
        # plt.close()

        # # 聚类的不确定性评分图保存
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(n_clusters), uncertainties, color='skyblue')
        # plt.xlabel('Cluster')
        # plt.ylabel('Uncertainty Score')
        # plt.title('Uncertainty Score by Cluster')
        # plt.xticks(range(n_clusters), [f'Cluster {i}' for i in range(n_clusters)])
        # plt.savefig('/root/YePeng/yolov5ds/111111/聚类的不确定性评分图.png')
        # plt.close()
        #
        # CONBINED = [image_uncertainty_score, number]
        # ADA.append(CONBINED)

        # #############聚类方法的可视化结果代码 也就是聚类的结果##############
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from sklearn.decomposition import PCA
        # from sklearn.cluster import SpectralClustering
        # from scipy.stats import entropy
        #
        #
        # # 假设 probabilities 是从模型预测得到的，维度为 12600x9
        # #np.random.seed(1)
        # probabilities = np.random.rand(12600, 9)  # 模拟数据
        #
        # # 提取特征：置信度和类别预测概率
        # features = probabilities[:, 4:]
        #
        # # 使用 SpectralClustering 进行谱聚类
        # n_clusters = 4  # 聚类的数量
        # spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10,
        #                               random_state=0)
        # labels = spectral.fit_predict(features)
        #
        # # 使用PCA进行降维以便于可视化
        # pca = PCA(n_components=2)
        # reduced_features = pca.fit_transform(features)
        #
        # # 可视化聚类结果
        # plt.figure(figsize=(10, 6))
        # for i in range(n_clusters):
        #     plt.scatter(reduced_features[labels == i, 0], reduced_features[labels == i, 1], label=f'Cluster {i}')
        # plt.title('Spectral Clustering Results')
        # plt.xlabel('PCA Feature 1')
        # plt.ylabel('PCA Feature 2')
        # plt.legend()
        # plt.grid(True)
        #
        # # 保存图像到指定文件夹
        # output_folder = '/root/YePeng/yolov5ds/cluster_results/'
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)
        # plt.savefig(f'{output_folder}/cluster1002_visualization.png')
        #
        # # 显示图像
        # plt.show()

        #---------------------------------------------------------------------------------------------#
        ### 瀵规瘮瀹為獙涓€鏄疪andom 闅忔満鎶芥牱 鍥犱负闅忔満鎶芥牱涓嶆秹鍙婃帹锟?鎵€浠ヨ繖閲屾垜浠殏鏃朵笉锟?        ## 瀵规瘮瀹為獙锟? Entropy Sampling
        ## 璁＄畻姣忎釜鏍锋湰鐨勭喌
        #entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=-1)  # 鍔犱笂涓€涓皬鍊间互閬垮厤log(0)
        #max_entropy_sample_index = np.argmax(entropies)
        #max_entropy_sample_entropy = entropies[max_entropy_sample_index]
        ##print(f"瀵瑰簲鐨凟ntropy Sampling锟? {max_entropy_sample_entropy}")
        ## 鑾峰彇姣忎竴寮犲浘鐗囬娴嬪€间腑鐨勭疆淇″害锛屽苟涓斿瓨鍒癆BC褰撲腑
        #CONBINED = [max_entropy_sample_entropy, number]
        #ADA.append(CONBINED)
        ##print(CONBINED)
        #---------------------------------------------------------------------------------------------#
        # # 瀵规瘮瀹為獙锟? Margin Sampling
        # # 杩欓噷宸茬煡鍓嶉潰鐨凙ctiveLearning_pred_cpu鍜屾鐜囧垎甯僷robabilities
        # # 璁＄畻姣忎釜鏍锋湰鐨勬渶澶ф鐜囧拰娆″ぇ姒傜巼
        # max_probs = np.max(probabilities, axis=-1)
        # print('1', max_probs)
        # second_max_probs = np.partition(probabilities, -2, axis=-1)[:, -2]
        # print('2',second_max_probs)
        # # 璁＄畻 Margin
        # margins = max_probs - second_max_probs
        # print('3',margins)
        # # 閫夋嫨 Margin 鏈€灏忕殑鏍锋湰
        # min_margin_sample_index = margins.count(0)
        # #min_margin_sample_index = np.argmin(margins)
        # print('4',min_margin_sample_index)
        # min_margin_sample_margin = np.round_(margins[min_margin_sample_index], decimals=6)
        # print('5',min_margin_sample_margin)
        # CONBINED = [min_margin_sample_margin, number]
        # ADA.append(CONBINED)
        #---------------------------------------------------------------------------------------------#
        # # 瀵规瘮瀹為獙锟? Least Confident
        # # 杩欓噷宸茬煡鍓嶉潰鐨凙ctiveLearning_pred_cpu鍜屾鐜囧垎甯僷robabilities
        # # 璁＄畻姣忎釜鏍锋湰鐨勬渶灏忔锟?
        # min_probs = np.min(probabilities, axis=-1)
        # #print('aaaaaaa', min_probs.shape)
        # # 閫夋嫨姒傜巼鏈€浣庣殑鏍锋湰
        # least_confident_sample_index = np.argmin(min_probs)
        # least_confident_sample_confidence = 1 - min_probs[least_confident_sample_index]  # 灏嗘渶灏忔鐜囪浆鎹负缃俊锟?        # #print(least_confident_sample_confidence)
        # CONBINED = [least_confident_sample_confidence, number]
        # ADA.append(CONBINED)
        # ##########接下来我们复现的是论文中的avg,max和sum的方法############
        # import numpy as np
        # # 假设这是预测后的概率数组，其中最后四列是四个不同类别的预测概率
        # # 我们用随机数生成一个模拟的概率数组来演示，实际中应使用您的预测数据
        # # probabilities = np.random.rand(12600, 9) # 示例用
        # # 为了演示，我们这里生成一个具有相同形状的随机数组
        #
        # # 提取四个类别的预测概率
        # class_probabilities = probabilities[:, 5:]
        # # 定义 v1vs2 函数
        # def v1vs2(class_probs):
        #     # 找到每个样本最高和次高的概率
        #     max_probs = np.max(class_probs, axis=1)
        #     second_max_probs = np.array([np.partition(pred, -2)[-2] for pred in class_probs])
        #     # 计算 v1vs2 指标
        #     v1vs2_values = (1 - max_probs - second_max_probs) ** 2
        #     return v1vs2_values
        # # 计算每个样本的 v1vs2 值
        # v1vs2_values = v1vs2(class_probabilities)
        # # 聚合指标的函数
        # def aggregate_values(v1vs2_values, method='sum'):
        #     if method == 'sum':
        #         return np.sum(v1vs2_values)
        #     elif method == 'avg':
        #         return np.mean(v1vs2_values)
        #     elif method == 'max':
        #         return np.max(v1vs2_values)
        #     else:
        #         raise ValueError("Invalid method specified")
        # #使用 'sum' 聚合方法 （高）
        # sum_aggregated_value = aggregate_values(v1vs2_values, method='sum')
        # print('111111', sum_aggregated_value)
        # CONBINED = [sum_aggregated_value, number]
        # ADA.append(CONBINED)
        # 使用 'avg' 聚合方法 （高）
        # # avg_aggregated_value = aggregate_values(v1vs2_values, method='avg')
        # # CONBINED = [avg_aggregated_value, number]
        # # ADA.append(CONBINED)
        # # # 使用 'max' 聚合方法 （高）
        # # max_aggregated_value = aggregate_values(v1vs2_values, method='max')
        # # CONBINED = [max_aggregated_value, number]
        # # ADA.append(CONBINED)

        #######边缘分割代码#########
        # import torch.nn.functional as F
        # def sobel_edge_detection_2d(input_2d):
        #     # Sobel算子核，需要在CPU或与输入相同的设备上
        #     sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=input_2d.device)
        #     sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=input_2d.device)
        #
        #     # 输入需要额外的批次和通道维度以适应conv2d的需求
        #     input_2d = input_2d.unsqueeze(0).unsqueeze(0)
        #
        #     # 应用Sobel算子
        #     edge_x = F.conv2d(input_2d, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        #     edge_y = F.conv2d(input_2d, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        #
        #     # 计算梯度幅值并移除额外的维度
        #     edge = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(0).squeeze(0)
        #
        #     return edge
        #
        # def calculate_uncertainty(probability_map):
        #     epsilon = 1e-7  # 微小的数值，用于数值稳定性处理
        #     # 使用 clamp 函数限制概率值的范围，避免接近 0 或 1
        #     probability_map = torch.clamp(probability_map, epsilon, 1.0 - epsilon)
        #     # 计算熵作为不确定性
        #     p_log_p = probability_map * torch.log(probability_map)
        #     entropy = -torch.sum(p_log_p, dim=1, keepdim=True)  # 沿类别维度求和
        #     return entropy
        #
        # endtime = time.time()
        # totaltime += (endtime - starttime)
        # ####################### segmentation post process #######################
        # segout = predout[1]
        # segout = torch.nn.functional.softmax(segout,dim=1)
        #
        # # 计算不确定性 新加的
        # uncertainty = calculate_uncertainty(segout) #新加的
        # nan_mask = torch.isnan(uncertainty)  # 创建一个布尔型张量，标记NaN位置为True
        # nan_count = torch.sum(nan_mask)  # 计算True的总数，即NaN的数量
        # print("Number of NaNs in uncertainty:", nan_count.item())
        #
        # print('uncertainty', uncertainty.shape)
        # segout = segout.squeeze(0)
        # mask = torch.argmax(segout,dim=0)
        #
        # # 应用Sobel算子进行边缘检测 新加的
        # B, C, H, W = uncertainty.size()
        # edges = sobel_edge_detection_2d(mask.float()) #新加的
        # # 检查uncertainty中是否有NaN值，并计算NaN的数量
        # nan_mask = torch.isnan(edges)  # 创建一个布尔型张量，标记NaN位置为True
        # nan_count1 = torch.sum(nan_mask)  # 计算True的总数，即NaN的数量
        # print("Number of NaNs in uncertainty:", nan_count1.item())
        # edges_expanded = edges.unsqueeze(0).unsqueeze(0)  # 增加批次和通道维度
        # edges_expanded = edges_expanded.expand(B, C, H, W)  # 扩展维度以匹配uncertainty
        #
        # print('edges_expanded', edges_expanded.shape)
        # # 将不确定性与边缘强度相乘以得到加权不确定性 #新加的
        # weighted_uncertainty = uncertainty * edges_expanded #新加的
        #
        #
        # # 加权不确定性的总和可以用于样本选择 #新加的
        # sample_uncertainty_score_sum = torch.sum(weighted_uncertainty, dim=(2,3))  # 沿宽高维度求和 #新加的
        # print(sample_uncertainty_score_sum)
        # # 计算像素点个数
        # num_pixels = weighted_uncertainty.size(2) * weighted_uncertainty.size(3)
        # # 对加权不确定性总和进行平均,可以选择对加权不确定性的总和进行平均，这样可以获得更平滑的结果，而不会受到图像尺寸的影响
        # sample_uncertainty_score = sample_uncertainty_score_sum / num_pixels
        # print(sample_uncertainty_score)
        # CONBINED = [sample_uncertainty_score, number] #新加的
        # ADA.append(CONBINED) #新加的

        segout = predout[1]
        segout = torch.nn.functional.softmax(segout, dim=1)



        # 假设 segout 是模型的预测输出，形状为 [batch_size, num_classes, height, width]
        # 首先，我们将 segout 通过 softmax 转换为概率分布
        segout = torch.nn.functional.softmax(segout, dim=1)

        # 定义一个函数来创建 Gabor 滤波器
        def gabor_kernel(frequency, theta, sigma_x, sigma_y, n_stds=3):
            # 生成 Gabor 滤波器的尺寸
            xmax = max(abs(n_stds * sigma_x * np.cos(theta)), abs(n_stds * sigma_y * np.sin(theta)))
            xmax = np.ceil(max(1, xmax))
            ymax = max(abs(n_stds * sigma_x * np.sin(theta)), abs(n_stds * sigma_y * np.cos(theta)))
            ymax = np.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

            # 计算 Gabor 滤波器
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
                2 * np.pi * frequency * x_theta)
            return gb.astype(np.float32)

        # 创建 Gabor 滤波器
        gabor_filter_numpy = gabor_kernel(frequency=0.25, theta=np.pi / 4, sigma_x=3, sigma_y=3)

        # 应用 Gabor 滤波器到图像的每个通道
        def gabor_edge_detector(image, kernel):
            # OpenCV 的 filter2D 函数需要输入是二维 NumPy 数组
            filtered_image = cv2.filter2D(image, cv2.CV_32F, kernel)
            return filtered_image

        # 选择单个图像的第一个通道
        # 通常我们会对灰度图像或单通道图像应用 Gabor 滤波器
        # 以下为了简化，我们只处理第一个类别的预测结果
        segout_single_channel_numpy = segout[0, 0].cpu().numpy()

        # 使用 Gabor 滤波器对单个通道图像进行边缘检测
        edge_detected = gabor_edge_detector(segout_single_channel_numpy, gabor_filter_numpy)

        # 因为边缘检测结果可能包含负值，我们将结果标准化到 [0, 1]
        edge_detected_normalized = (edge_detected - edge_detected.min()) / (edge_detected.max() - edge_detected.min())

        print('edge', edge_detected_normalized.shape)
        exit()
        # 公式(1)：为了简化，我们假设 k=1，即3x3区域
        def region_impurity(pred, k=1):
            # Calculate the region impurity for each pixel
            impurity_map = torch.zeros(pred.shape[2:], device=pred.device)
            for i in range(pred.shape[2]):
                for j in range(pred.shape[3]):
                    # Extract 3x3 neighborhood predictions
                    neighbor_pred = pred[:, :, max(i - k, 0):min(i + k + 1, pred.shape[2]),
                                    max(j - k, 0):min(j + k + 1, pred.shape[3])]
                    # Calculate the mean prediction within the region
                    mean_pred = torch.mean(neighbor_pred, dim=(2, 3))
                    # Compute the impurity (entropy)
                    mean_pred_clamped = torch.clamp(mean_pred, min=1e-7, max=1)
                    impurity = -torch.sum(mean_pred_clamped * torch.log(mean_pred_clamped),
                                          dim=1)  # Adding epsilon to avoid log(0)
                    impurity_map[i, j] = impurity
            return impurity_map

        # 公式(2) 和 公式(6)：我们先计算预测不确定性
        def prediction_uncertainty(pred):
            # 确保概率值位于 [1e-10, 1] 范围内，避免对数操作的数值问题
            pred_clamped = torch.clamp(pred, min=1e-7, max=1.0)
            # 计算每个像素点的预测熵
            uncertainty_map = -torch.sum(pred_clamped * torch.log(pred_clamped), dim=1)
            return uncertainty_map

        # 公式(6)：计算最终的获取函数
        def final_acquisition_function(impurity, uncertainty, alpha=0.5):
            # 假设我们使用简单的加权平均来组合不纯净度和不确定性
            # 确保uncertainty的维度与impurity一致
            if uncertainty.dim() == 3 and uncertainty.shape[0] == 1:
                uncertainty = uncertainty.squeeze(0)

            # 确保impurity的维度与uncertainty一致
            if impurity.dim() == 2 and len(uncertainty.shape) == 3:
                impurity = impurity.unsqueeze(0)

            # # 计算最终的获取函数
            # return alpha * impurity + (1 - alpha) * uncertainty

            # 计算最终的获取函数
            return impurity * uncertainty

        # 执行函数
        impurity_map = region_impurity(segout)
        # nan_count1 = torch.isnan(impurity_map).sum().item()
        # print(f"Number of NaNs in uncertainty_map: {nan_count1}")
        # print(impurity_map.shape)
        uncertainty_map = prediction_uncertainty(segout)
        acquisition_function = final_acquisition_function(impurity_map, uncertainty_map)
        acquisition_function =  acquisition_function * edge_detected_normalized
        acquisition_function_numpy = acquisition_function.cpu().numpy()
        # 打印形状确认结果
        print(acquisition_function_numpy)
        # 假设 acquisition_function 是一个 PyTorch 张量且位于 GPU 上
        # 首先将其移动到 CPU 并转换为 NumPy 数组
        #acquisition_function_numpy = acquisition_function.cpu().numpy()

        # # 找到具有最高获取分数的10个像素点
        # top10_indices = np.unravel_index(np.argsort(acquisition_function_numpy.ravel())[-100:],
        #                                  acquisition_function_numpy.shape)
        # print(top10_indices)
        #
        # # 获取这10个像素点的分数
        # top10_scores = acquisition_function_numpy[top10_indices]
        # print(top10_scores)
        # # 计算这10个分数的均值
        # top10_mean_score = np.mean(top10_scores)
        # print(top10_mean_score)
        # print("Mean of top 10 acquisition scores:", top10_mean_score)

        # 计算整个张量的平均获取分数
        average_acquisition_score = np.mean(acquisition_function_numpy)
        CONBINED = [average_acquisition_score, number] #新加的
        ADA.append(CONBINED) #新加的
        #print("Average acquisition score for the image:", average_acquisition_score)
        # 你可能想要选择具有最高获取分数的像素点或区域
        # 例如，选择分数最高的10个像素点
        # indices = np.unravel_index(np.argsort(acquisition_function_numpy.ravel())[-10:],
        #                            acquisition_function_numpy.shape)
        # selected_pixels = list(zip(indices[0], indices[1]))
        # print("Selected pixels for annotation:", selected_pixels)


        segout = segout.squeeze(0)
        mask = torch.argmax(segout, dim=0)
        # print("segout",segout.shape, mask)
        mask = mask.detach().cpu().numpy()
        mask = mask.astype(np.uint8)
        oldshape = mask.shape
        mask = mask[int(pad[1]):int(oldshape[0]-pad[1]), int(pad[0]):int(oldshape[1]-pad[0])]
        #print('!!!!!!!!!!!!!:', mask.shape)
        mask1 = cv2.resize(mask,(1024,512)) ###hanfujun
        #mask1 = cv2.resize(mask,(1280,720)) ###hanfujun
        #print(mask1)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        pred_color = colorEncode(mask, colorsmap).astype(np.uint8)
        # mask = mask*20
        #cv2.imwrite("mask.jpg",pred_color)
        t3 = time_sync()
        dt[1] += t3 - t2
        # print("out shape : ",type(predout),len(predout),predout[0][0].shape, predout[1].shape, im.shape)
        # NMS
        pred = non_max_suppression(predout[0][0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

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
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 杩欓噷鏄敾妗嗭紝浣嗘槸鍚庨潰鎴戜滑浼氫慨鏀筶abel鍦ㄧ敾涓€锟?鎵€浠ヨ繖涓€娆″氨鎶婁粬娉ㄩ噴浜嗭紝閬垮厤鐢讳袱娆★拷?                        #annotator.box_label(xyxy, label, color=colors(c, True))
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
                        #if label_split[1] == 'Vehicle,Bicycle':
                           #label = yes_or_no[idx] + ',' + 'T,B'
                           #label = yes_or_no[idx] + ',' + 'Bicycle'
                        #elif label_split[1] == 'Vehicle,Motorcycle':
                           #label = yes_or_no[idx] + ',' + 'T,M'
                           #label = yes_or_no[idx] + ',' + 'Motorcycle'
                        #if label_split[0] == 'Pedestrian':
                           #label = yes_or_no[idx] + ',' + 'P,P'
                           #label = yes_or_no[idx]
                        #elif label_split[0] == 'Person,Rider':
                           #label = yes_or_no[idx] + ',' + 'P,R'
                           #label = yes_or_no[idx] + ',' + label
                           #label = yes_or_no[idx] + ',' + 'Rider'
                        if label_split[0] == 'Bicycle':
                           #label = yes_or_no[idx]
                           label = yes_or_no[idx] + ',' + 'B'
                           #label = yes_or_no[idx] + ',' + 'Bicycle'
                        elif label_split[0] == 'Motorcycle':
                           #label = yes_or_no[idx]
                           label = yes_or_no[idx] + ',' + 'M'
                           #label = yes_or_no[idx] + ',' + 'Motorcycle'
                        elif label_split[0] == 'Pedestrian':
                           #label = yes_or_no[idx]
                           label = yes_or_no[idx] + ',' + 'P'
                           #label = yes_or_no[idx] + ','+'Pedestrian'
                        elif label_split[0] == 'Rider':
                           #label = yes_or_no[idx]
                           label = yes_or_no[idx] + ',' + 'R'
                           #label = yes_or_no[idx] + ',' + 'Rider'
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
                  #  im_vis = im0 * 0.7 + maskcolor * 0.3
                 #   im_vis = im_vis.astype(np.uint8)
                    im_vis = cv2.addWeighted(im0,0.5,maskcolor,0.5,1); # 绗竴绡囨枃绔犺繖閲屽垎鍒槸0.6 锟?0.4
                    cv2.imwrite("{}_yolods.png".format(save_path[:-4]),im_vis)
               #     cv2.imwrite("{}_mask.png".format(save_path[:-4]), maskcolor)
               #     cv2.imwrite("{}_det.png".format(save_path[:-4]), im0)
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

    
    
    my_2d_list = ADA
    sorted_2d_list = sorted(my_2d_list, key=lambda x: x[0], reverse=True)
    print(sorted_2d_list)
    second_dimension_elements = [item[1] for item in sorted_2d_list]
    #print(second_dimension_elements)
    total_images = 2502     
    ratio = 0.01 
    number_quchu = int(total_images * ratio)   
    first_two_elements = second_dimension_elements[:number_quchu]
    print(first_two_elements)
    file_path = "/root/YePeng/yolov5ds/baocunID.txt"    
    with open(file_path, "w", encoding="utf-8") as file:
        for element in first_two_elements:
            file.write(element + "\n")



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
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '/root/YePeng/P3ADA/runs/Normal-->Foggy/Source only/exp/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='/root/YePeng/yolov5ds/datasetbianhao/images/', help='file/dir/URL/glob, 0 for webcam')
    #parser.add_argument('--source', type=str, default='/root/YePeng/yolov5ds/images1/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold') # 鎭跺姡澶╂皵閫夋嫨0.25 姝ｅ父澶╂皵閫夋嫨0.5
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
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
