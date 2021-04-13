import torch
from collections import Counter
from Intersection_over_Union import intersection_over_union


def mean_avg_precision(
    pred_boxes,true_boxes,iou_threshold=0.5,box_format="corners",num_classes=20
):
    avg_precisions=[]
    epsilon=1e-6

    for c in range(num_classes):
        detections=[]
        ground_truths=[]

        for detection in pred_boxes:
            if detection[1]==c:
                detections.append(detection)
            
            for true_box in true_boxes:
                if true_box[1]==c:
                    ground_truths.append(true_box)
            

            amount_bbox=Counter([gt[0] for gt in ground_truths ])

            for key,val in amount_bbox.items():
                amount_bbox[key]=torch.zeros(val)
            
            detections.sort(key=lambda x:x[2],reverse=True)
            TP=torch.zeros((len(detections)))
            FP=torch.zeros((len(detections)))
            total_true_bboxes=len(ground_truths)
            for detection_idx,detection in enumerate(detections):
                ground_trut_img=[
                    bbox for bbox in ground_truths if bbox[0] == detection[0]   
                ]

                num_gts=len(ground_trut_img)
                best_iou=0

                for idx,gt in enumerate(ground_trut_img):
                    iou=intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format
                    )
                    if iou > best_iou:
                        best_iou =iou
                        best_gt_idx=idx
            if best_iou > iou_threshold:
                if amount_bbox[detection[0]][best_gt_idx]==0:
                    TP[detection_idx]=1
                    amount_bbox[detection[0]][best_gt_idx]=1
                else:
                    FP[detection_idx]=1
            else:
                FP[detection_idx]=1
        TP_CUMSUM=torch.cumsum(TP,dim=0)
        FP_CUSUM=torch.cumsum(FP,dim=0)
        recalls=TP_CUMSUM/(total_true_bboxes+epsilon)
        precisions=torch.divide(TP_CUMSUM,(TP_CUMSUM+FP_CUSUM+epsilon))
        precisions=torch.cat((torch.tensor([1]),precisions))
        recalls=torch.cat((torch.tensor([0]),recalls))
        avg_precisions.append(torch.trapz(precisions,recalls))

    return sum(avg_precisions) /len(avg_precisions)
