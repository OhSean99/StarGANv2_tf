import time

## for Data
dataBasePath = "./DB/celeba_hq/train"
domainList = ["female", "male"]
imageShape = (256, 256, 3)
doAugmentation = True

## for BASEMODEL
imgSize = 256
imgCh = 3
numDomains = 2
numStyle = 5
maxConvDim = 512
sn = False
styleDim = 64
hiddenDim = 512
latentDim = 16

## for BASEMODEL.train
learningRate = 1e-4
learningRate_f = 1e-6
beta1 = 0.0
beta2 = 0.99
emaDecay = 0.999

iteration_S = 0
iteration_E = 100000
batchSize = 12

ganType = "gan-gp"
dsWeight = 1
dsIter = 100000
adversarialWeight = 1
r1Weight = 1
styleWeight = 1
cycleWeight = 1

modelPath = "./testModel1"
loggingName = f'train_{time.strftime("%Y%m%d-%H%M%S")}'
loggingFolder = './logs'
loggingSnapshotInterval = 1
validationFolder = './validation'