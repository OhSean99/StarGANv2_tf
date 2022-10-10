from basemodel import BASEMODEL
from dataset import Data
import tensorflow as tf

from arguments.train_args import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

if __name__ == "__main__":
  
  trainData = Data(dataBasePath = dataBasePath,
                  domainList = domainList,
                  imageShape = imageShape,
                  doAugmentation = doAugmentation)

  model = BASEMODEL(True,
                    imgSize = imgSize,
                    imgCh = imgCh,
                    numDomains = numDomains,
                    numStyle = numStyle,
                    maxConvDim = maxConvDim,
                    sn = sn,
                    styleDim = styleDim,
                    hiddenDim = hiddenDim,
                    latentDim = latentDim)

  model.train(learningRate = learningRate,
              learningRate_f = learningRate_f,
              beta1 = beta1,
              beta2 = beta2,
              emaDecay = emaDecay,

              iteration_S = iteration_S,
              iteration_E = iteration_E,
              batchSize = batchSize,

              ganType = ganType,
              dsWeight = dsWeight,
              dsIter = dsIter,
              adversarialWeight = adversarialWeight,
              r1Weight = r1Weight,
              styleWeight = styleWeight,
              cycleWeight = cycleWeight,

              modelPath = modelPath,
              loggingName = loggingName,
              loggingFolder = loggingFolder,
              loggingSnapshotInterval = loggingSnapshotInterval,
              validationFolder = validationFolder
              )
