export save='logs/rsn101_coco'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_multi.lua \
	-data '/home/yluo/project/dataset/mscoco/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'coco' \
	-resetClassifier true \
	-nClasses 160 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-lFunc 'mce' | tee $save/log.txt
export save='logs/rsn101_coco_gsm'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_multi.lua \
	-data '/home/yluo/project/dataset/mscoco/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'coco' \
	-resetClassifier true \
	-nClasses 160 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-gsm_mu 0 \
	-gsm_sigma 1 \
	-gsm_scale 1 \
	-lFunc 'gsm' | tee $save/log.txt