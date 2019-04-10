--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train_multi'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

if opt.lFunc == 'mce' then
   dofile('src/MultiCrossEntropyCriterion.lua')
elseif opt.lFunc == 'gsm' then
   dofile('src/MltGaussianSoftMaxCriterion.lua')
end

paths.mkdir(opt.save)
paths.mkdir(paths.concat(opt.save,'results'))
testLogger = optim.Logger(paths.concat(opt.save,'test.log'))
testLogger:setNames{'TrainLoss','TestLoss'}
testLogger.showPlot = false

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model and criterion
local model, criterion = models.setup(opt, checkpoint)
if criterion == nil then
   if opt.lFunc == 'mce' then
      criterion = nn.MultiCrossEntropyCriterion():type(opt.tensorType)
   elseif opt.lFunc == 'gsm' then
      criterion = nn.MltGaussianSoftMaxCriterion(opt.nClasses/2,opt.gsm_mu,opt.gsm_sigma):type(opt.tensorType)
   end
end

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

print((' | # of training images: %d, # of val images: %d'):format(trainLoader.__size,valLoader.__size))
print(opt)

if opt.testOnly then
   trainer:test(0, valLoader)
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestErr = 0 --math.huge
local bestEp = -1
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testLoss = trainer:test(epoch, valLoader)

   if testLogger then
      testLogger:add{trainLoss, testLoss}
      testLogger:style{'+-','+-'}
      testLogger:plot()
   end

   local bestModel = false
   if testLoss < bestErr then
      bestErr = testLoss
      bestEp = epoch
      bestModel = true
      print((' * Best model at %d-th epoch'):format(epoch))
   end

   --checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
   checkpoints.save_all(epoch, model, criterion, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished min Loss: %6.4f at %d-th epoch', bestErr, bestEp))
