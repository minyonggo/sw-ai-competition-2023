########################################
########## model setting ###############
########################################

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # Segmentation usually uses SyncBN

crop_size = (256, 256)  # The crop size during training.
test_img_size = (224, 224)
data_preprocessor = dict(  # The config of data preprocessor, usually includes image 
    type='SegDataPreProcessor',  # The type of data preprocessor.
    mean=[87.3359904, 91.29594915, 83.01340125],
    std=[43.7584692, 38.600545950000004, 35.4384822],
    bgr_to_rgb=True,  # Whether to convert image from BGR to RGB.
    pad_val=0,  # Padding value of image.
    seg_pad_val=255,  # Padding value of segmentation map.
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,  # The ImageNet pretrained backbone to be loaded
    backbone=dict(
        type='ResNetV1c',  # The type of backbone
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,   # The configuration of norm layer.
        norm_eval=False,  # Whether to freeze the statistics in BN
        style='pytorch',
        contract_dilation=True),  # When dilation > 1, whether contract first layer of dilation.

    decode_head=dict(
        type='DepthwiseSeparableASPPHead',  # Type of decode head
        in_channels=2048,  # Input channel of decode head.
        in_index=3,  # The index of feature map to select.
        channels=512,  # The intermediate channels of decode head.
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,  # The dropout ratio before final classification layer.
        num_classes=2,  # Number of segmentation class.
        norm_cfg=norm_cfg,
        align_corners=False,  # The align_corners argument for resize in decoding.
        loss_decode=dict(  # Type of loss used for segmentation.
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,  # Input channel of auxiliary head.
        in_index=2,  # The index of feature map to select.
        channels=256,  # The intermediate channels of decode head.
        num_convs=1,  # Number of convs in FCNHead. It is usually 1 in auxiliary head.
        concat_input=False,  # Whether concat output of convs with input before classification layer.
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # The test mode, options are 'whole' and 'slide'


########################################
########## dataset setting #############
########################################


# dataset settings
dataset_type = 'SatelliteDataset'  # Dataset type, this will be used to define the dataset.
data_root = 'data/Satellite'  # Root path of data.
# crop_size = (512, 512)  # The crop size during training.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Second pipeline to load annotations for current image.

    # Augmentation pipeline that resize the images and their annotations.
    # dict(
    #     type='RandomResize',
    #     scale=(1024, 1024),
    #     ratio_range=(0.7, 1.3),
    #     keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
    dict(type='Resize', scale=test_img_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
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
########## schedule 80k ################
########################################

# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=30000,
        by_epoch=False)
]

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=250))



########################################
########## default runtime #############
########################################

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'

# LoveDA - DeepLabV3+, R-101-D8
load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_loveda/deeplabv3plus_r101-d8_512x512_80k_loveda_20211105_110759-4c1f297e.pth"

tta_model = dict(type='SegTTAModel')