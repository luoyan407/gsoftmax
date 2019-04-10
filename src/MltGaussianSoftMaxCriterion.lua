--[[
    Gaussian-based criterion for multi-label classification
--]]
require 'distributions'
require 'src/MultiCrossEntropyCriterion'

local MltGaussianSoftMaxCriterion, parent = torch.class('nn.MltGaussianSoftMaxCriterion', 'nn.Criterion')

--[[
    Function: this function constructs a criterion object
    Inputs
    bzsize: the number of classes
    ini_mu: the initialized value of mu
    ini_sigma: the initialized value of sigma
    scale: the weight controlling contribution of distribution term
    Output(s) return a criterion object
--]]
function MltGaussianSoftMaxCriterion:__init(bzsize, ini_mu, ini_sigma, scale)
    parent.__init(self)
    self.mu = ini_mu or 0
    self.mu = torch.ones(bzsize,2)*self.mu
    self.sigma = ini_sigma  or 1
    self.sigma = torch.ones(bzsize,2)*self.sigma
    self.cecriterion = nn.MultiCrossEntropyCriterion()
    self.scale = scale or 1
end

local eps = math.exp(-51*math.log(2))

--[[
    Function: forward process of the Gaussian-based softmax criterion
    Inputs
    _input: the output features of the fully-connected layer
    _target: an one-hot vector representing the image's groundtruth labels
    Output(s) return a loss
--]]
function MltGaussianSoftMaxCriterion:updateOutput(_input,_target)
    local input = _input:clone()
    input = torch.reshape(input, input:size(1), input:size(2)/2, 2)
    
    local prob = torch.zeros(input:size())
    for i=1,prob:size(2) do
        prob[{ {},i,1 }] = distributions.norm.cdf(input[{ {},i,1 }]:double(), self.mu[{ i,1 }], self.sigma[{ i,1 }])
        prob[{ {},i,2 }] = distributions.norm.cdf(input[{ {},i,2 }]:double(), self.mu[{ i,2 }], self.sigma[{ i,2 }])
    end
    prob = (prob:typeAs(input))*self.scale + input
    prob = torch.reshape(prob, prob:size(1), prob:size(2)*prob:size(3))
    self.output, preds, preds_mask = self.cecriterion:updateOutput(prob,_target)

    return self.output, preds, preds_mask
end

--[[
    Function: back-propagation to generate the gradient w.r.t. the input
    Inputs
    _input: the output features of the fully-connected layer
    _target: an one-hot vector representing the image's groundtruth labels 
    Output(s) return gradients
--]]
function MltGaussianSoftMaxCriterion:updateGradInput(_input,_target)
    local input = _input:clone()
    input = torch.reshape(input, input:size(1), input:size(2)/2, 2)
    
    local prob = torch.zeros(input:size())
    for i=1,prob:size(2) do
        prob[{ {},i,1 }] = distributions.norm.cdf(input[{ {},i,1 }]:double(), self.mu[{ i,1 }], self.sigma[{ i,1 }])
        prob[{ {},i,2 }] = distributions.norm.cdf(input[{ {},i,2 }]:double(), self.mu[{ i,2 }], self.sigma[{ i,2 }])
    end
    prob = (prob:typeAs(input))*self.scale + input
    prob = torch.reshape(prob, prob:size(1), prob:size(2)*prob:size(3))
    local sfGradInput = self.cecriterion:updateGradInput(prob,_target)

    local ext_mu = torch.repeatTensor(self.mu, input:size(1),1,1)
    local ext_sigma = torch.repeatTensor(self.sigma, input:size(1),1,1)

    local gradInput = torch.exp(-torch.pow(input-ext_mu,2):cdiv(2*torch.pow(ext_sigma,2)))
    gradInput = torch.cdiv(gradInput,torch.sqrt(2*math.pi)*ext_sigma)
    gradInput = gradInput*self.scale
    gradInput = torch.view(gradInput, _input:size(1),_input:size(2))
    gradInput = gradInput:add(1):cmul(sfGradInput)

    self.gradInput = gradInput
    return self.gradInput
end

--[[
    Function: this function updates parameters
    Inputs
    _input: the output features of the fully-connected layer
    _target: an one-hot vector representing the image's groundtruth labels 
    lr:     the learning rate
--]]
function MltGaussianSoftMaxCriterion:updateParameters(_input,_target,lr)
    local input = _input:clone()
    input = torch.reshape(input, input:size(1), input:size(2)/2, 2)
    
    local prob = torch.zeros(input:size())
    for i=1,prob:size(2) do
        prob[{ {},i,1 }] = distributions.norm.cdf(input[{ {},i,1 }]:double(), self.mu[{ i,1 }], self.sigma[{ i,1 }])
        prob[{ {},i,2 }] = distributions.norm.cdf(input[{ {},i,2 }]:double(), self.mu[{ i,2 }], self.sigma[{ i,2 }])
    end
    prob = (prob:typeAs(input))*self.scale + input
    prob = torch.reshape(prob, prob:size(1), prob:size(2)*prob:size(3))
    local sfGradInput = self.cecriterion:updateGradInput(prob,_target)
    sfGradInput = torch.view(sfGradInput, input:size())

    local ext_mu = torch.repeatTensor(self.mu, input:size(1),1,1)
    local ext_sigma = torch.repeatTensor(self.sigma, input:size(1),1,1)

    local gradInput = torch.exp(-torch.pow(input-ext_mu,2):cdiv(2*torch.pow(ext_sigma,2)))
    gradInput = gradInput*self.scale

    local dzdmu = torch.cdiv(gradInput,torch.sqrt(2*math.pi)*ext_sigma)
    dzdmu = dzdmu:cmul(sfGradInput)
    local dzdsigma = torch.cdiv(gradInput,torch.sqrt(2*math.pi)*torch.pow(ext_sigma,2)):cmul(ext_mu-input)
    dzdsigma = dzdsigma:cmul(sfGradInput)

    dzdmu = torch.mean(dzdmu, 1):squeeze()
    dzdsigma = torch.mean(dzdsigma, 1):squeeze()

    self.mu = self.mu-lr*dzdmu
    self.sigma = self.sigma-lr*dzdsigma

    return self.mu, self.sigma
end

--[[
    Function: this function prints out the criterion
--]]
function MltGaussianSoftMaxCriterion:__tostring__()
    local s = string.format('%s%s', torch.type(self), self:printParameters())
    return s
end

--[[
    Function: this function predicts the labels according the features
    Inputs
    _input: the output features of the fully-connected layer
--]]
function MltGaussianSoftMaxCriterion:predict(_input)
    local input = _input:clone()

    local prob = torch.zeros(_input:size())
    for i=1,prob:size(2) do
        prob[{ {},i }] = distributions.norm.cdf(input[{ {},i }]:double(), self.mu[i], self.sigma[i])
    end
    prob = (prob:typeAs(input))*self.scale + input
    local exp_sum = torch.exp(prob):sum(2):repeatTensor(1,input:size(2))
    local output = torch.exp(prob):cdiv(exp_sum)

    return output
end