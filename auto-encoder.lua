#!/usr/bin/env lorch

require 'nngraph'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

function loadDataset(fn)
  local f = torch.load(fn, 'ascii')
  -- binarize the data
  return f.data:type(torch.getdefaulttensortype()):gt(256/2):float()
end

function batchDataset(data, batchSize)
  local nExample = math.floor(data:size(1) / batchSize)
  local perm = torch.randperm(data:size(1)):long()
  return data:index(1,perm):narrow(1,1,nExample*batchSize):view(nExample,batchSize,-1)
end

function train(model, data)
  -- epoch tracker
  epoch = epoch or 1
  iterStartedAt = torch.tic()

  -- local vars
  local runningBound = nil
  local examplesProcessed = 0
  local B = batchSize
  local ones = torch.ones(batchSize):mul(-1)
  local maxNorm = maxNorm or 10

  -- do one epoch
  local dataset = batchDataset(data, B)
  for t=1,dataset:size(1) do
    collectgarbage()
    -- We just use x, not the labels.
    local x = dataset[t]

    -- create closure to evaluate f(w) and df/dW
    local feval = function(w)
      -- get new parameters
      if w ~= model.par then
        model.par:copy(w)
      end
      -- reset gradients
      model.gradPar:zero()

      -- sample noise variates
      local e = torch.randn(batchSize,zSize)
      -- evaluate function for complete mini batch
      local elob = model.net:forward({e,x})
      -- estimate db/dW
      local df_do = model.net:backward({e,x},ones)[1]

      if runningBound == nil then
        runningBound = elob
      else
        runningBound = runningBound * 0.95 + elob * 0.05
      end

      local updateNorm = model.gradPar:norm()
      if updateNorm > maxNorm then
        model.gradPar:div(updateNorm/maxNorm)
      end

      return elob, model.gradPar
    end

    adamConfig = adamConfig or {
      learningRate = 0.001,
      weightDecay = 0,
    }
    optim.adam(feval, model.par, adamConfig)

    -- diplay progress
    if t % 20 == 0 then
      print("[" .. epoch .. ',' .. t .. "] " .. runningBound:mean() .. ", norm: "
        .. model.par:norm() .. ", gradNorm: " .. model.gradPar:norm())
    end
  end
  -- next epoch
  epoch = epoch + 1
end

function makeNet(xSize, zSize, hSize)
  local prior = nn.Sequential()
  prior:add(nn.Linear(zSize, hSize))
  prior:add(nn.Tanh())
  prior:add(nn.Linear(hSize,xSize))
  prior:add(nn.Sigmoid())

  local qMu = nn.Sequential()
  qMu:add(nn.Linear(xSize, hSize))
  qMu:add(nn.Tanh())
  qMu:add(nn.Linear(hSize,zSize))

  local qSigma = nn.Sequential()
  qSigma:add(nn.Linear(xSize, hSize))
  qSigma:add(nn.Tanh())
  qSigma:add(nn.Linear(hSize,zSize))

  local ns = {}
  -- Takes noise, e, and datapoint, x
  ns.inTuple = nn.Identity()()
  ns.e = nn.SelectTable(1)(ns.inTuple)
  ns.x = nn.SelectTable(2)(ns.inTuple)
  -- Compute parameters of approximate posterior.
  ns.qMu = qMu(ns.x)
  ns.qSigma = qSigma(ns.x)
  ns.qVar = nn.Power(2)(ns.qSigma)
  -- Reparameterization trick.
  ns.z = nn.CAddTable()({nn.CMulTable()({ns.e, ns.qSigma}), ns.qMu})
  -- KL-divergence part
  ns.kl = nn.MulConstant(0.5)(nn.Sum(1,1)(nn.CAddTable()({
    nn.AddConstant(1)(nn.Log()(ns.qVar)),
    nn.MulConstant(-1)(nn.Power(2)(ns.qMu)),
    nn.MulConstant(-1)(ns.qVar),
  })))
  -- Compute the log-likelihood of the data.
  ns.y = prior(ns.z)
  ns.likelihood = nn.CAddTable()({
    nn.CMulTable()({ns.x, nn.Log()(ns.y)}),
    nn.CMulTable()({
      nn.AddConstant(1)(nn.MulConstant(-1)(ns.x)),
      nn.Log()(nn.AddConstant(1)(nn.MulConstant(-1)(ns.y)))
    })
  })
  -- Put together the variational evidence lower bound
  ns.elob = nn.CAddTable()({ns.kl, nn.Sum(1,1)(ns.likelihood)})
  -- Make a module out of the whole graph.
  ns.net = nn.gModule({ns.inTuple},{ns.elob})
  -- Vectorize our parameters
  ns.par, ns.gradPar = ns.net:getParameters()
  return ns
end

-- Read in the data.
trainingFn = "mnist.t7/train_32x32.t7"
trainingData = loadDataset(trainingFn)

-- Set the parameters
batchSize = 100
xSize = 32*32
hSize = 500
zSize = 3

-- Check the gradient for a small model
small = makeNet(9, 2, 3)
x1 = torch.Tensor(2,9):uniform():gt(0.5):float()
e1 = torch.randn(2,2)
one = torch.Tensor(1):fill(1)
sum = nn.Sum()

local gradCheck = function(x)
  small.par:copy(x)
  small.gradPar:zero()
  local out = small.net:forward({e1,x1}) 
  local obj = sum:forward(out)
  local back = sum:backward(out, one)
  small.net:backward({e1,x1}, back)
  return obj, small.gradPar
end
local randTheta = torch.Tensor():resizeAs(small.par):uniform(-0.1, 0.1)
local a,b,c = optim.checkgrad(gradCheck, randTheta, 1e-04)
print("gradient error: " .. a)

-- Make the full network
model = makeNet(xSize, zSize, hSize)

-- Manual computation
print("Manually checking net computation")
x1 = batchDataset(trainingData, batchSize)[1]
e1 = torch.randn(batchSize,zSize)
b1 = model.net:forward({e1,x1})
mu1 = model.qMu.data.module:forward(x1)
sigma1 = model.qSigma.data.module:forward(x1)
var1 = torch.pow(sigma1,2)
z1 = e1:cmul(sigma1):add(mu1)
y1 = model.y.data.module:forward(z1)
kl = (torch.log(var1) + 1 - torch.pow(mu1,2) - var1):sum() / 2
ll = torch.ones(batchSize, xSize)
ll:add(-1, x1)
ll:cmul(torch.ones(batchSize,xSize):add(-1, y1):log())
ll:add(torch.log(y1):cmul(x1))
b2 = kl + ll:sum()
print("net produced: " .. b1:sum() .. ", manual computation: " .. b2)

-- Train the network
model.par:uniform(-0.1, 0.1)
epoch = 1
for k=1,20 do
  train(model, trainingData)
  if k % 5 == 0 then
    torch.save("trained-model.t7", model)
  end
end

-- END
