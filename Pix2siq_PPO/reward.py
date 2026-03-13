# reward model
import torch
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy


class RecallReward(nn.Module):
    def __init__(self, IoU_thresh=None, duplicate_penalty=0.3, num_bins=2000, num_classes=91, image_size=1333):
        # recall reward model in the paper
        # reward = number of matched ground truth boxes
        #          - duplicate_penalty * number of duplicate boxes
        super(RecallReward, self).__init__()
        self.IoU_thresh = IoU_thresh
        self.duplicate_penalty = duplicate_penalty
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.image_size = image_size

    def forward(self, inputs, targets):
        # inputs: outputs from actor model
        # targets: dataset ground truth
        out_seq_logits = inputs['pred_seq_logits']
        #b_labels = torch.stack([t['labels'] for t in targets], dim=0)
        #b_boxes = torch.stack([t['boxes'] for t in targets], dim=0)
        assert len(out_seq_logits) == len(targets), 'batch size should be equal to targets!'
        rewards = torch.zeros(len(out_seq_logits))
        # print(rewards)
        for b_i, pred_seq_logits in enumerate(out_seq_logits):
            labels = targets[b_i]['labels']
            boxes = targets[b_i]['boxes']
            #orig_size = targets[b_i]['orig_size']
            input_size = targets[b_i]['size']
            #ori_img_h, ori_img_w = orig_size
            inp_img_h, inp_img_w = input_size
            #scale_fct_w = ori_img_w / inp_img_w
            #scale_fct_h = ori_img_h / inp_img_h
            # print('ground truth cxcywh:', boxes)
            boxes = box_cxcywh_to_xyxy(boxes)
            # print('ground truth xyxy:', boxes)
            seq_len = pred_seq_logits.shape[0]
            pred_seq_logits = pred_seq_logits.softmax(dim=-1)
            # print(pred_seq_logits.shape)
            num_objects = seq_len // 5
            pred_seq_logits = pred_seq_logits[:int(num_objects * 5)].reshape(num_objects, 5, -1)
            pred_boxes_logits = pred_seq_logits[:, :4, :self.num_bins + 1]
            pred_class_logits = pred_seq_logits[:, 4, self.num_bins + 1: self.num_bins + 1 + self.num_classes]
            _, pred_classes = torch.max(pred_class_logits, dim=1)
            pred_bbox = pred_boxes_logits.argmax(dim=2) * self.image_size / self.num_bins
            pred_bbox[:, 0::2] /= inp_img_w
            pred_bbox[:, 1::2] /= inp_img_h
            pred_bbox[pred_bbox > 1] = 1
            pred_bbox[pred_bbox < 0] = 0
            #pred_bbox = torch.clamp(pred_bbox, min=0, max=1)
            # print(pred_classes)
            # print(pred_bbox)
            # print(labels)
            reward = 0
            for label, box in zip(labels, boxes):
                inds = pred_classes==label
                pred_i_bbox = pred_bbox[inds]
                if len(pred_i_bbox) == 0:
                    continue
                # print('pred:', pred_i_bbox)
                # print('gt:', box)
                IoUs = self.compute_IoU(pred_i_bbox, box)
                # print('IoUs:', IoUs)
                if self.IoU_thresh is not None:
                    bingos = (IoUs > self.IoU_thresh).sum()
                else:
                    thresh = 0.5
                    bingos = 0
                    while thresh < 1:
                        bingos += ((IoUs > thresh).sum() * 0.1)
                        thresh += 0.05
                # print('bingos:', bingos)
                if bingos > 1:
                    reward += max((1 - self.duplicate_penalty * (bingos - 1)), 0)
                else:
                    reward += bingos
                # print(reward)
            if len(labels) > 0:
                reward = reward / len(labels)
            #    rewards[b_i] = 0
            #else:
            #    rewards[b_i] = reward / len(labels)
            rewards[b_i] = reward * 100 + 10
            # print('rewards:', rewards)
        #rewards = rewards.mean()
        return rewards

    def compute_IoU(self, bboxes, target_box):
        # bboxes: [[x0, y0, x1, y1], ...] within 0~1
        # target_box: target bbox [x0, y0, x1, y1] within 0~1
        # return IoUs: [IoU_1, IoU_2, IoU_3, ...]
        target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
        # print('target_area', target_area)
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        left_top = torch.max(bboxes[:, :2], target_box[:2])
        right_bottom = torch.min(bboxes[:, 2:], target_box[2:])
        intersection = torch.clamp(right_bottom - left_top, min=0)
        #print('intersection', intersection)
        intersection_areas = intersection[:, 0] * intersection[:, 1]
        #print('intersection_areas', intersection_areas)
        union_areas = torch.clamp((bbox_areas + target_area) - intersection_areas, min=1e-10)
        IoUs = intersection_areas / union_areas
        return IoUs
    


def build_reward_model(model_type):
    if model_type == 'recall':
        return RecallReward()
    raise ValueError('{} reward model not implenmented!'.format(model_type))


if __name__ == '__main__':
    print('='*40)
    print('Reward model test.')
    print('='*40)
    import argparse
    from torch.utils.data import Dataset, DataLoader
    from playground import build_all_model
    from common import get_args_parser
    from datasets.coco import CocoDetection
    import datasets.transforms as T
    import util.misc as utils
    from ppo import make_coco_transforms

    # setup model
    parser = argparse.ArgumentParser('Pix2Seq PPO script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    model, criterion, postprocessors = build_all_model[args.model](args)
    model.to(device)
    if not args.resume:
        raise RuntimeError('Please specify model weight path using --resume')
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # setup dataset
    img_folder = 'coco/train2017'
    ann_file = 'coco/annotations/instances_train2017.json'
    dataset_train = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms('ppo'),
        return_masks=False,
        large_scale_jitter=False,
        image_set='train'
    )
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 1, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, num_workers=2)
    
    # test
    reward_model = RecallReward()
    reward_model.to(device)
    for sample, target in data_loader_train:
        # print(target)
        sample = sample.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]
        out = model([sample, target])# ['pred_seq_logits']
        #print(out[0].shape)
        rewards = reward_model(out, targets)
        break
