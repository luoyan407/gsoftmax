--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

require 'paths'
local optim = require 'optim'
local matio = require 'matio'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

local function freezeLayers(model, layerNames)
   for i=1,#layerNames do
      local layerName = layerNames[i]
      model:apply(function(m)
         if torch.type(m):find(layerName) then
            m:evaluate()
         end
      end)
   end
   return model
end

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   self.isLogged = opt.isLogged
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local totalPred = nil
   local totalPredLabel = nil
   local totalLabel = nil
   local totalFeat = nil

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0
   local lossSum = 0.0
   local hitNumers = nil
   local filenames = {}

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local needReshape = false
      if output:size():size() == 1 then
         needReshape = true
      end
      local batchSize = self.input:size(1)
      if needReshape then
         output = torch.view(output, self.input:size(1), output:size(1)/self.input:size(1))
         self.model.output = torch.view(self.model.output, self.input:size(1), self.model.output:size(1)/self.input:size(1))
      end
      local feat = torch.view(output,output:size(1),output:size(2)/2,2)
      local loss,preds,pred_labels = self.criterion:forward(self.model.output, self.target)

      if not totalPred then
         totalPred = preds
         totalPredLabel = pred_labels
         totalFeat = feat
      else
         totalPred = totalPred:cat(preds,1)
         totalPredLabel = totalPredLabel:cat(pred_labels,1)
         totalFeat = totalFeat:cat(feat,1)
      end
      if not totalLabel then
         totalLabel = sample.target
      else
         totalLabel = totalLabel:cat(sample.target,1)
      end

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      if torch.type(self.criterion)=='nn.MltGaussianSoftMaxCriterion' then
         self.criterion:updateParameters(self.model.output, self.target, self.optimState.learningRate*self.opt.gsm_lr_w)
      end
      if needReshape then
         self.criterion.gradInput = torch.view(self.criterion.gradInput, self.criterion.gradInput:size(1)*self.criterion.gradInput:size(2))
      end
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())
      
      dataTimer:reset()
   end

   print((' | Epoch-Training: [%d]  Err %1.4f  time %.2fmins'):format(
         epoch, lossSum / N, timer:time().real/60))
   timer:reset()

   matio.save(paths.concat(self.opt.save,'results','ep_train_'..string.format('%03d',epoch)..'.mat'),
               {predProb=totalPred:float(), predLabel=totalPredLabel:float(), gtLabel=totalLabel:float(), feat=totalFeat:float()})

   return lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local totalPred = nil
   local totalPredLabel = nil
   local totalLabel = nil
   local totalFeat = nil

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0
   local lossSum = 0.0
   local hitNumers = nil
   local filenames = {}
   local feat2048,feat160 = nil,nil

   nClass = 1000
   local gtRecord = torch.zeros(nClass)
   local resRecord = torch.zeros(nClass)
   local resRecord_img = {}
   local fFile = nil
   if self.isLogged then
      fFile = assert(io.open(paths.concat(self.opt.save, 'val_img_epoch'..epoch), 'w'))
   end

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local needReshape = false
      if output:size():size() == 1 then
         needReshape = true
      end
      if needReshape then
         output = torch.view(output, self.input:size(1), output:size(1)/self.input:size(1))
         self.model.output = torch.view(self.model.output, self.input:size(1), self.model.output:size(1)/self.input:size(1))
      end
      local batchSize = self.input:size(1) / nCrops
      local feat = torch.view(output,output:size(1),output:size(2)/2,2)
      local loss,preds,pred_labels = self.criterion:forward(self.model.output, self.target)

      if not totalPred then
         totalPred = preds
         totalPredLabel = pred_labels
         totalFeat = feat
      else
         totalPred = totalPred:cat(preds,1)
         totalPredLabel = totalPredLabel:cat(pred_labels,1)
         totalFeat = totalFeat:cat(feat,1)
      end
      if not totalLabel then
         totalLabel = sample.target
      else
         totalLabel = totalLabel:cat(sample.target,1)
      end

      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      if fFile then
         for i=1,#sample.imgnames do
            local tmpRecord = string.format('%s %d %d\n',sample.imgnames[i], batch_mAP[i], batch_mAP10[i])
            fFile:write(tmpRecord)
         end
      end

      dataTimer:reset()
   end
   self.model:training()

   if fFile then
      fFile:close()
   end

   print((' | Epoch-Test: [%d]  Err %1.4f  time %.2fmins'):format(
         epoch, lossSum / N, timer:time().real/60))
   timer:reset()

   matio.save(paths.concat(self.opt.save,'results','ep_test_'..string.format('%03d',epoch)..'.mat'),
               {predProb=totalPred:float(), predLabel=totalPredLabel:float(), gtLabel=totalLabel:float(), feat=totalFeat:float()})

   return lossSum / N
end

function Trainer:computeMuSigma(dataloader)
   local size = dataloader:size()
   local feat_out = nil
   self.model:evaluate()
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
      if not feat_out then
         feat_out = output
      else
         feat_out = torch.cat(feat_out, output, 1)
      end

   end
   self.model:training()

   feat_out = torch.reshape(feat_out, feat_out:size(1), feat_out:size(2)/2, 2)
   self.criterion.mu = torch.mean(feat_out, 1):squeeze():typeAs(self.criterion.mu)
   local ext_mu = torch.repeatTensor(self.criterion.mu, feat_out:size(1), 1, 1)
                        :typeAs(self.criterion.mu)
   self.criterion.sigma = torch.pow(feat_out-ext_mu,2):mean(1):squeeze():pow(0.5)
                              :typeAs(self.criterion.sigma)
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():topk(5, 2, true, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(predictions))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100, correct:narrow(2, 1, 1):clone()
end

local function getCudaTensorType(tensorType)
  if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
  elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
  else
     return cutorch.createCudaHostTensor()
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'coco' then
      if epoch > 3 then
         decay = 1
      elseif epoch > 6 then
         decay = 2
      end
   elseif self.opt.dataset == 'nuswideclean' then
      if epoch > 3 then
         decay = 1
      elseif epoch > 6 then
         decay = 2
      end
   else
      if epoch > 3 then
         decay = 1
      end
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
