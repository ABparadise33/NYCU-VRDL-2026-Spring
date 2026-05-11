_base_ = "./cascade_mask_rcnn_r50_fpn_hw3_adamw_cosine.py"

train_dataloader = dict(
    dataset=dict(ann_file="annotations_adamw85/instances_hw3_train.json")
)

val_dataloader = dict(
    dataset=dict(ann_file="annotations_adamw85/instances_hw3_val.json")
)

val_evaluator = dict(ann_file="./annotations_adamw85/instances_hw3_val.json")
