# Deepstream_LPD_LPR

<h3>Steps to Run </h3>

1- Download the tlt-tool according cuda,cudnn and tensorrt version read the readme on that converter folder<br>
Ref_link: https://docs.nvidia.com/tlt/tlt-user-guide/text/tensorrt.html#installing-the-tlt-converter <br>
Convert the model into engine file using command <br>
$ ./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 ./us_lprnet_baseline18_deployable.etlt -t fp16 -e ./lpr_us_onnx_b16.engine<br>

2-Place this engine file in /models/LPR/<br>

3.Make .so file <br>
$ cd nvinfer_custom_lpr_parser <br>
$ make <br>

3-Run the Deepstream app<br>
$ cd LPR_On_Images<br>
$ python3 only_lpr_classifier.py <file_name.jpg><br>
e.g:<br>
$ python3 only_lpr_classifier.py ./index.jpeg<br>

