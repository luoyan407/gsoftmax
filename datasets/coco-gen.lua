local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' and class ~= '.DS_Store' then
         table.insert(classList, class)
         classToIdx[class] = #classList
      end
   end

   return classList, classToIdx
end

local function findImages(dir, classToIdx)
   local imagePath = torch.CharTensor()
   
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   local classdim = -1

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)

      local path = filename
      if classdim < 0 then
         classdim = classToIdx[filename]:size(1)
      end

      table.insert(imagePaths, path)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   local imageClass = torch.IntTensor(#imagePaths, classdim)
   for i,v in ipairs(imagePaths) do
      local classId = classToIdx[v]:int()
      imageClass[{ i,{1,classdim} }] = classId
   end

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   --local imageClass = torch.LongTensor(imageClasses)
   return imagePath, imageClass
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = paths.concat(opt.data, 'train2014')
   local valDir = paths.concat(opt.data, 'val2014')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   local tmpobj
   tmpobj = torch.load(opt.data..'val2014.t7')
   local valImageMapping = tmpobj.imageToClassIdx
   tmpobj = torch.load(opt.data..'train2014.t7')
   local trainImageMapping = tmpobj.imageToClassIdx
   local classList = tmpobj.classList
   local classToIdx = tmpobj.classToIdx

   print("=> Generating list of images")
   --local classList, classToIdx = findClasses(trainDir)

   print(" | finding all validation images")
   local valImagePath, valImageClass = findImages(valDir, valImageMapping)

   print(" | finding all training images")
   local trainImagePath, trainImageClass = findImages(trainDir, trainImageMapping)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
