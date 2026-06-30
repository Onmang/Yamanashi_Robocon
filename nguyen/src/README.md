#jetson nano env.

### env
GPU = Tegra X1/ 4GB
Jetpack == 4.6
ubuntu == 18.04
python == 3.8
numpy == 1.23.5
onnx gpu -> "https://elinux.org/Jetson_Zoo#ONNX_Runtime"
ultralytics -> "https://i7y.org/en/yolov8-on-jetson-nano/"
tensorrt -> "sudo apt install nvidia-tensorrt"

### nvcc PATHを通す

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version

