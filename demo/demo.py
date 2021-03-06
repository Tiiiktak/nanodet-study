import cv2
import os
import time
import torch
import argparse
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='video', help='demo type, eg. image, video and webcam')
    parser.add_argument('--config', help='model config file path',
                        default='/home/dsh/icecream_project_data/model_checkpoints/nanodet_stage2_0330_config.yml')
    parser.add_argument('--model', help='model file path',
                        default='/home/dsh/icecream_project_data/model_checkpoints'
                                '/nanodet_stage2_0407_epoch26_model_best.pth')
    parser.add_argument('--path', help='path to images or video',
                        default='/home/dsh/icecream_project_data/2021-03-17_13-23-47_CBQKL+HFL.avi')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    parser.add_argument('--device', type=str, default='cpu', help='device cpu or cuda:x')
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == 'RepVGG':
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({'deploy': True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))

    def draw_bbox(self, dets, meta, class_names, score_thres):
        time1 = time.time()
        result = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=False)
        print('viz time: {:.3f}s'.format(time.time() - time1))
        return result


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main():
    args = parse_args()
    if args.device != 'cpu':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(-1, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device=args.device)
    logger.log('Press "Esc", "q" or "Q" to exit.')
    if args.demo == 'image':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            predictor.visualize(res, meta, cfg.class_names, 0.35)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
        result_video_path = args.path.replace('.avi', '_result.avi')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (w, h)
        result_cap = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                break
            meta, res = predictor.inference(frame)
            # predictor.visualize(res, meta, cfg.class_names, 0.35)
            result_frame = predictor.draw_bbox(res, meta, cfg.class_names, 0.36)
            write_frame = cv2.resize(result_frame, (w, h), interpolation=cv2.INTER_NEAREST)
            result_cap.write(write_frame)
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord('q') or ch == ord('Q'):
            #     break


if __name__ == '__main__':
    main()
