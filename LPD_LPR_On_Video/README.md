# Deepdtream_LPD_LPR

Steps to Run 
1-Download LPRNet model and Place in /models/LPR/cuda11.1_cudnn8.0_trt7.2-20210304T191646Z-001/cuda11.1_cudnn8.0_trt7.2
Download link - https://ngc.nvidia.com/catalog/models/nvidia:tlt_lprnet
NOTE: Download deployable_v1.0

2-Convert the model into engine file using command
./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 ./us_lprnet_baseline18_deployable.etlt -t fp16 -e ./lpr_us_onnx_b16.engine
NOTE: download the tlt-tool according cuda,cudnn and tensorrt version read the readme on that converter folder

3-Place this engine file in /models/LPRNet/

4-Run the Deepstream app
python3 deepstream_lpr_app.py <video_file>
