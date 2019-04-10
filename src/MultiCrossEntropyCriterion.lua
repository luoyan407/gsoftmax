--[[
    Gaussian-based criterion for multi-label classification
--]]
local MultiCrossEntropyCriterion, parent = torch.class('nn.MultiCrossEntropyCriterion', 'nn.Criterion')

function MultiCrossEntropyCriterion:__init()
    parent.__init(self)
end

local eps = math.exp(-51*math.log(2))

function MultiCrossEntropyCriterion:updateOutput(_input,_target)
    local input = _input:clone()
    local target = _target:clone()

    input = torch.reshape(_input, _input:size(1),_input:size(2)/2,2) -- reshape
    local rev_target = target-1
    rev_target = torch.mul(rev_target,-1)
    target = torch.cat(target,rev_target,3)
    local max_input = torch.max(input, 3):
                        repeatTensor(1,1,2)
    input = input-max_input
    input = torch.exp(input)
    local sum_input = torch.sum(input, 3):
                        repeatTensor(1,1,2)

    input = torch.cdiv(input,sum_input)
    local output = torch.log(input+eps):cmul(-target:typeAs(input))
    local preds_mask = torch.ge(input[{ {},{},1 }],input[{ {},{},2 }]):typeAs(input)
    local preds = input[{ {},{},1 }]

    self.output = output:sum()/input:size(1)
    return self.output, preds, preds_mask
end

--input dim: b*1*h*w
--output dim: b*1*h*w
function MultiCrossEntropyCriterion:updateGradInput(_input,_target)
    local input = _input:clone()
    local target = _target:clone()

    input = torch.reshape(_input, _input:size(1),_input:size(2)/2,2) -- reshape
    local rev_target = target-1
    rev_target = torch.mul(rev_target,-1)
    target = torch.cat(target,rev_target,3)
    local max_input = torch.max(input, 3):
                        repeatTensor(1,1,2)
    input = input-max_input
    input = torch.exp(input)
    local sum_input = torch.sum(input, 3):
                        repeatTensor(1,1,2)
    input = torch.cdiv(input,sum_input)
    local gradInput = input-target:typeAs(input)
    gradInput = torch.view(gradInput, _input:size(1),_input:size(2))

    self.gradInput = gradInput
    return self.gradInput
end

