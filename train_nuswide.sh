export save='logs/rsn101_nuswide'
mkdir -p $save
CUDA_VISIBLE_DEVICES=0,1 th main_multi.lua \
	-data '/home/yluo/project/dataset/nuswide/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'nuswideclean' \
	-resetClassifier true \
	-nClasses 162 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-lFunc 'mce' | tee $save/log.txt
export save='logs/rsn101_nuswide_gsm'
mkdir -p $save
CUDA_VISIBLE_DEVICES=0,1 th main_multi.lua \
	-data '/home/yluo/project/dataset/nuswide/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'nuswideclean' \
	-resetClassifier true \
	-nClasses 162 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-gsm_mu 0 \
	-gsm_sigma 1 \
	-gsm_scale .1 \
	-gsm_lr_w 100 \
	-lFunc 'gsm' | tee $save/log.txt