local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local NUSWIDECleanDataset = torch.class('resnet.NUSWIDECleanDataset', M)

function str2array(input)
   out = {}
   tokens = input:split(" ")
   for i=1,#tokens do
      out[i] = tonumber(tokens[i])
   end
   return out
end

function NUSWIDECleanDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data

   self.targetSize = opt.imgSize
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function NUSWIDECleanDataset:get(i)
   local path = self.imageInfo.imagePath[i]

   local filename = path

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[{ i,{} }]

   return {
      input = image,
      target = class,
      filename = filename,
      salmap = nil,
   }
end

function NUSWIDECleanDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function NUSWIDECleanDataset:_loadSalImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   return ok, input
end

function NUSWIDECleanDataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function NUSWIDECleanDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ScaleWH(self.targetSize),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.ScaleWH(self.targetSize),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.NUSWIDECleanDataset
