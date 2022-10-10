
ckptPath = r"C:\Users\sangi\Documents\repos\face_gender_swap\testModel1\ckpt"
baseDataPath = r"C:\Users\sangi\Documents\repos\face_gender_swap\DB\celeba_hq\val"
imgSize = 256
imgCh = 3

targetDomain = 1 # 0 = male to female, 1 = female to male
numSources = 5
numGenerations = 5
outPath = r"C:\Users\sangi\Documents\repos\face_gender_swap\testModel1\testOut"

## for BASEMODEL
numDomains = 2
numStyle = 5
maxConvDim = 512
sn = False
styleDim = 64
hiddenDim = 512
latentDim = 16