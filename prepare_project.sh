conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https:/github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .