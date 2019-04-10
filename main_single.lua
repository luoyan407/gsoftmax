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
local DataLoader = require 'dataloader_single'
local models = require 'models/init'
local Trainer = require 'train'
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

if opt.lFunc == 'gsm' then
   dofile('src/GaussianSoftMaxCriterion.lua')
end

paths.mkdir(opt.save)
paths.mkdir(paths.concat(opt.save,'results'))
testLogger = optim.Logger(paths.concat(opt.save,'test.log'))
testLogger:setNames{'Train_Top1','Train_Top5','Train_Loss','Test_Top1','Test_Top5','Test_Loss'}
testLogger.showPlot = false

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model and criterion
local model, criterion = models.setup(opt, checkpoint)
if opt.lFunc == 'gsm' then
   criterion = nn.GaussianSoftMaxCriterion(opt.nClasses, opt.gsm_mu, opt.gsm_sigma, opt.gsm_scale):type(opt.tensorType)
else
   criterion = nn.CrossEntropyCriterion():type(opt.tensorType)
end

--print(model)
print(opt)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local test_top1, test_top5, test_loss = trainer:test(0, valLoader)
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestErr = 100 --math.huge
local bestEp = -1
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local train_top1, train_top5, train_loss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local test_top1, test_top5, test_loss = trainer:test(epoch, valLoader)

   if testLogger then
      testLogger:add{train_top1, train_top5, train_loss, test_top1, test_top5, test_loss}
      testLogger:style{'+-','+-','+-','+-','+-','+-'}
      testLogger:plot()
   end

   local bestModel = false
   if test_top1 < bestErr then
      bestErr = test_top1
      bestEp = epoch
      bestModel = true
      print((' * Best model at %d-th epoch Top 1 error %6.4f'):format(bestEp, bestErr))
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished Top 1 error: %6.4f at %d-th epoch', bestErr, bestEp))
