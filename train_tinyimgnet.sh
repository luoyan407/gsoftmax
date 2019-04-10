export save='logs/rsn101_tinyimagenet'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_single.lua \
	-data '/home/yluo/project/dataset/tinyimagenet/images' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 80 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'tinyimagenet' \
	-resetClassifier true \
	-nClasses 200 \
	-LR 1e-03 \
	-imgSize 224 \
	-featSize 7 \
	-nEpochs 30 \
	-lFunc 'ce' | tee $save/log.txt
export save='logs/rsn101_tinyimagenet_gsm'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_single.lua \
	-data '/home/yluo/project/dataset/tinyimagenet/images' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 80 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'tinyimagenet' \
	-resetClassifier true \
	-nClasses 200 \
	-LR 1e-03 \
	-imgSize 224 \
	-featSize 7 \
	-nEpochs 30 \
	-gsm_mu 0 \
	-gsm_sigma 1 \
	-gsm_scale .1 \
	-lFunc 'gsm' | tee $save/log.txt