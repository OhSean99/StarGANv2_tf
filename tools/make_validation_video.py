import os
import cv2
import tqdm

basePath = r"C:\Users\sangi\Documents\repos\face_gender_swap\validation"
outPath = r"C:\Users\sangi\Documents\repos\face_gender_swap\testModel1"

latImages, refImages = [], []

for fn in os.listdir(basePath):
  if "latent" in fn:
    latImages.append(os.path.join(basePath, fn))
  elif "ref" in fn:
    refImages.append(os.path.join(basePath, fn))

latImages = sorted(latImages, key = lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1]))
refImages = sorted(refImages, key = lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1]))

for (ii, latfn) in tqdm.tqdm(enumerate(latImages), total = len(latImages)):
  img = cv2.imread(latfn, -1)

  if ii == 0:
    outcap = cv2.VideoWriter(os.path.join(outPath, f"training_latent_synth.mp4"),
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30,
                            (img.shape[1], img.shape[0]), True)

  iter0 = os.path.splitext(os.path.basename(latfn))[0].split("_")[1]
  cv2.putText(img, f"training iterations: {iter0}", (10, img.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 1, ( 0, 0, 0), 1)
  outcap.write(img)
outcap.release()

for (ii, reffn) in tqdm.tqdm(enumerate(refImages), total = len(refImages)):
  img = cv2.imread(reffn, -1)

  if ii == 0:
    outcap = cv2.VideoWriter(os.path.join(outPath, f"training_reference_synth.mp4"),
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30,
                            (img.shape[1], img.shape[0]), True)

  iter0 = os.path.splitext(os.path.basename(reffn))[0].split("_")[1]
  cv2.putText(img, f"training iterations: {iter0}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, ( 0, 0, 0), 1)
  outcap.write(img)
outcap.release()
