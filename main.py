from mmdet.apis import DetInferencer
from mmengine.config import Config
import cv2
from tools.dataset_converters.mot2coco import CLASSES

# Initialize the DetInferencer
CUSTOM_CLASSES = {
        'classes':
        ('person'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }
cfg = Config.fromfile('mmdetection/mmdet/configs/rtmdet/rtmdet_tiny_8xb32_300e_coco.py')
cfg.model.bbox_head.num_classes = len(CUSTOM_CLASSES)  # 클래스 수를 커스텀 클래스 개수로 설정
cfg.classes = CUSTOM_CLASSES  # 커스텀 클래스를 설정
cfg.data.test.pipeline[1].dataset.classes = CUSTOM_CLASSES  # 테스트 데이터 파이프라인에 반영

# 수정된 설정을 기반으로 DetInferencer 초기화
inferencer = DetInferencer(cfg)

# 추론 수행
inferencer('./test_data/img.png', show=True)