from mmdet.apis import DetInferencer
# Initialize the DetInferencer
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# Perform inference
inferencer('./test_data/img.png', show=True)