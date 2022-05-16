import numpy as np
import cv2

def RootSIFT_create():
    return RootSIFT()

class RootSIFT:
 
    def detectAndCompute(self, image, unnecessary_param=None, eps=1e-7):
        # initialize the SIFT feature extractor
        sift = cv2.SIFT_create()
        # extractor = cv2.SiftDescriptorExtractor
        
        # detect Difference of Gaussian keypoints in the image
        # detector = cv2.SiftFeatureDetector()
        kps = sift.detect(image)
        # extract normal SIFT descriptors
        (kps, descs) = sift.compute(image, kps)
        # print("SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))
        # compute SIFT descriptors
        (kps, descs) = sift.compute(image, kps)
        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        # return a tuple of the keypoints and descriptors
        return (kps, descs)