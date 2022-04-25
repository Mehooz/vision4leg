ID=$1
SEED=$2
LOGDIR=$3

python3 convert_locotransformer_half.py --log_dir $3 --id $ID --seed $SEED

sudo /usr/src/tensorrt/bin/trtexec --onnx=onnx/${ID}_seed${SEED}_half.onnx --saveEngine=trt_engine/${ID}_seed${SEED}_fp16.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16

sudo cp -r trt_engine/${ID}_seed${SEED}_fp16.trt ~/sliding_image/trt_engine/