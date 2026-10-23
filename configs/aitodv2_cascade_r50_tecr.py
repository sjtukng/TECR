"""Copy this file to mmdet-nwdrka/configs_nwdrka/tecr/ in the AI-TOD fork."""

_base_ = '../baseline/aitodv2_cascade_rcnn_r50_baseline_1x.py'

custom_imports = dict(imports=['tecr_v2_core.neck'], allow_failed_imports=False)

model = dict(neck=dict(
    type='TECRV2FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
    enhanced_levels=(0, 1),
    lambda_init=0.1,
    normalize_eps=1e-6,
))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=1)
