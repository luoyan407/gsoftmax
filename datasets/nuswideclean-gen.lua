require 'io'
local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function file_exists(file)
   local f = io.open(file, 'rb')
   if f then f:close() end
   return f~=nil
end

local function lines_from_tensor(file, valids, num_images, num_classes)
   total_valids = 0
   for k, v in pairs(valids) do
      total_valids = total_valids + v
   end
   assert(total_valids == num_images, ('mis-matching images and labels %d vs %d'):format(total_valids, num_images))
   local labels = torch.IntTensor(num_images, num_classes):fill(0)
   if not file_exists(file) then
      return nil
   end
   n_lines = 0
   n_valids = 0
   for line in io.lines(file) do
      n_lines = n_lines + 1
      if valids[n_lines] == 1 then
         n_valids = n_valids + 1
         local tokens = line:split(" ")
         for i=1,#tokens do
            labels[{ n_valids,i }] = tonumber(tokens[i])
         end
      end
   end
   return labels
end

local function get_paths(file)
   local lines = {}
   local valids = {}
   if not file_exists(file) then
      return lines
   end
   local validlen
   local n_lines = 0
   for line in io.lines(file) do
      n_lines = n_lines + 1
      valids[n_lines] = 0
      local tokens = line:split(" ")
      local img_path = string.sub(tokens[1], string.find(tokens[1],'/')+1, string.len(tokens[1]))
      if tonumber(tokens[2]) > 0 then
         lines[#lines+1] = img_path
         valids[n_lines] = 1
      end
   end
   return lines, valids
end

local function lines_from(file)
   local lines = {}
   if not file_exists(file) then
      return lines
   end
   local validlen
   for line in io.lines(file) do
      validlen = string.len(line)
      if string.byte(string.sub(line,string.len(line),string.len(line)))==13 then
         validlen = validlen - 1
      end
      lines[#lines+1] = string.sub(line, 1, validlen)
   end
   return lines
end

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


   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
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

   return imagePath, imageClass
end

function M.exec(opt, cacheFile)
   print("=> Generating list of images")
   local trainImgList = paths.concat(opt.data,'..','clean','nus_wide_train_imglist.txt')
   local valImgList = paths.concat(opt.data,'..','clean','nus_wide_test_imglist.txt')
   local classListFile = paths.concat(opt.data,'..','Concepts81.txt')
   local trainLabelFile = paths.concat(opt.data,'..','clean','nus_wide_train_label.txt')
   local testLabelFile = paths.concat(opt.data,'..','clean','nus_wide_test_label.txt')

   local classList = lines_from(classListFile)
   print(" | finding all images")
   local trainImagePath, trainValids = get_paths(trainImgList)
   local valImagePath, valValids = get_paths(valImgList)
   local trainImageClass = torch.IntTensor(#trainImagePath, #classList):fill(0)
   local valImageClass = torch.IntTensor(#valImagePath, #classList):fill(0)
   local oneLabelFile, tmpIn, tmpTensor
   print(" | finding all classes")
   local trainLabelList = lines_from_tensor(trainLabelFile, trainValids, #trainImagePath, #classList)
   local testLabelList = lines_from_tensor(testLabelFile, valValids, #valImagePath, #classList)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainLabelList,
      },
      val = {
         imagePath = valImagePath,
         imageClass = testLabelList,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
