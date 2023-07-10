########################################
########## model setting ###############
########################################

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # Segmentation usually uses SyncBN

crop_size = (224, 224)  # The crop size during training.
data_preprocessor = dict(  # The config of data preprocessor, usually includes image 
    type='SegDataPreProcessor',  # The type of data preprocessor.
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,  # Whether to convert image from BGR to RGB.
    pad_val=0,  # Padding value of image.
    seg_pad_val=255,  # Padding value of segmentation map.
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',  # The ImageNet pretrained backbone to be loaded

    backbone=dict(
        type='BEiT',
        img_size=crop_size,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1),
        
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),

    decode_head=dict(
        type='UPerHead',  # Type of decode head
        in_channels=[768, 768, 768, 768],  # Input channel of decode head.
        in_index=[0, 1, 2, 3],  # The index of feature map to select.
        pool_scales=(1, 2, 3, 6),
        channels=768,  # The intermediate channels of decode head.
        dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        num_classes=2,  # Number of segmentation class.
        norm_cfg=norm_cfg,
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=dict(  # Type of loss used for segmentation.
            type='CrossEntropyLoss', use_sigmoid=False, class_weight=[0.25, 0.75], loss_weight=1.0)),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,  # Input channel of auxiliary head.
        in_index=2,  # The index of feature map to select.
        channels=256,  # The intermediate channels of decode head.
        num_convs=1,  # Number of convs in FCNHead. It is usually 1 in auxiliary head.
        concat_input=False,  # Whether concat output of convs with input before classification layer.
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, class_weight=[0.25, 0.75], loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)),   # The test mode, options are 'whole' and 'slide'
    )


########################################
########## dataset setting #############
########################################


# dataset settings
dataset_type = 'SatelliteDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/Satellite'  # Root path of data.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Second pipeline to load annotations for current image.

    # Augmentation pipeline that resize the images and their annotations.
    # dict(
    #     type='RandomResize',
    #     scale=(1024, 1024),
    #     ratio_range=(0.7, 1.3),
    #     keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.85),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]


train_dataloader = dict(
    batch_size=32,  # Batch size of a single GPU
    num_workers=4,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val_slice', seg_map_path='ann_dir/val_slice'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/test'),
        pipeline=test_pipeline))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])

# TODO: test evaluator for competition
test_evaluator = dict(
    type='CityscapesMetric',
    format_only=True,
    keep_results=True,
    output_dir='_satellite/deeplabv3plus/format_results')



########################################
############## scheduler ###############
########################################

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        power=1.0,
        begin=0,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=500))



########################################
########## default runtime #############
########################################

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
                # dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'

# load pretrained model from mmseg
load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-base_8x2_640x640_160k_ade20k/upernet_beit-base_8x2_640x640_160k_ade20k-eead221d.pth"

tta_model = dict(type='SegTTAModel')