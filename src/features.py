import math

import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.ndimage import filters, gaussian_filter
from cv2 import warpAffine
from cv2 import INTER_LINEAR
from scipy.spatial import distance

def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2],dtype=float)
        orientationImage = np.zeros(srcImage.shape[:2],dtype=float)

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        # sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        # sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

        sobx = np.zeros(srcImage.shape[:2],dtype=float)
        filters.sobel(srcImage,1,sobx)
        soby = np.zeros(srcImage.shape[:2],dtype=float)
        filters.sobel(srcImage,0,soby)
        # sobx = filters.convolve(srcImage,sx,mode='reflect')
        # soby = filters.convolve(srcImage,sy,mode='reflect')
        Ix = sobx*sobx
        Iy = soby*soby
        Ixy = sobx*soby


        Wxx = filters.gaussian_filter(Ix,sigma=0.5)
        Wyy = filters.gaussian_filter(Iy,sigma=0.5)
        Wxy = filters.gaussian_filter(Ixy,sigma=0.5)


        # for i in range(height):
        #     for j in range(width):
        #         M = np.array([[Wxx[i,j],Wxy[i,j]],[Wxy[i,j],Wyy[i,j]]])
        #         R = np.linalg.det((M)-0.1*np.trace(M)*np.trace(M))
        #         harrisImage[i,j] = R
        #         orientationImage[i, j] = np.arctan2(Ix[i, j], Iy[i, j]) * (180) / np.pi
                # orientationImage[i,j] = np.arctan2(Ix[i,j],Iy[i,j])
        harrisImage = Wxx*Wyy - Wxy*Wxy - 0.1*(Wxx+Wyy)*(Wxx+Wyy)
        orientationImage  = np.arctan2(soby,sobx)*(180) / np.pi
        # TODO-BLOCK-END

         # raise Exception("TODO in features.py not implemented")

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def checkBorder(self, va, borderA, vb, borderB):
            if va - 1 >= 0 and va + 1 < borderA and vb - 1 >= 0 and vb + 1 < borderB:
                return True
            else:
                return False



    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
                         :type harrisImage: object
        '''
        height, width = harrisImage.shape[:2]
        destImage = np.zeros_like(harrisImage, np.bool)

        # newpd = np.zeros((height+6,width+6),dtype=float)
        # newpd[3:3+height,3:3+width] = harrisImage
        # newmax = np.zeros(height,width)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        newmax = ndimage.maximum_filter(harrisImage,size=7)
        for i in range(height):
            for j in range(width):
                # newmax[i,j] = np.max(newpd[i:i+7,j:j+7])
                if harrisImage[i,j]==newmax[i,j]:
                    destImage[i,j] = True
                else:
                    destImage[i,j] = False
        # for y in range(height):
        #     for x in range(width):
        #         destImage[y,x] = True
        #         for j in range(-3,4):
        #             for i in range(-3,4):
        #                 if 0<=y+i<height and 0<=x+j<width and harrisImage[y+i,x+j]>harrisImage[y, x]:
        #                     destImage[y, x] = False

        # TODO-BLOCK-END
        # raise Exception("TODO in features.py not implemented")
        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.

        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                f = cv2.KeyPoint()
                f.size = 10
                f.angle = orientationImage[y,x]
                f.pt = (x,y)
                f.response = harrisImage[y,x]
                features.append(f)


                # TODO-BLOCK-BEGIN



                # raise Exception("TODO in features.py not implemented")
                # TODO-BLOCK-END


        return features

    def Adaptive_NonMaximal_Suppression(self,features):
        finalfeatures = []
        n = len(features)
        m = 500

        harrisvalues = []

        for f in features:
            harrisvalues.append(f.response)

        hmax = np.max(harrisvalues)
        crobust = 0.9

        r = np.zeros(n)
        Idx = -1

        for f1 in features:
            Idx+=1

            x1,y1 = f1.pt
            if(f1.response > crobust*hmax):
                r[Idx] = float("inf")

            else:
                di = []
                for f2 in features:
                    if f2.response > crobust*hmax or f2.response > crobust * f1.response:
                        continue

                    if f1 == f2:
                        continue

                    x2,y2 = f2.pt
                    dis = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    di.append(dis)

                r[Idx] = np.max(di)

        decIdx = np.argsort(-r)

        for j in range(m):
            finalfeatures.append(features[decIdx[j]])

    def ANMS(self,features):
        harrisvalues = []
        for f in features:
            harrisvalues.append(f.response)
        hmax = np.max(harrisvalues)

        decIdx = np.argsort(-harrisvalues)

        n = len(features)
        m = 500
        crobust = 0.9
        R = 100
        finalfeatures = []
        s = set()

        for i in range(5):
            for i in range(len(decIdx)):
                Index = decIdx[i]
                if(Index in s):
                    continue

                if features[Index].response > crobust*hmax:
                    finalfeatures.append(features[Index])
                    continue

                flag =True

                for f2 in features:
                    x1,y1 = features[Index].pt
                    x2,y2 = f2.pt

                    if(f2.response > crobust*features[Index] and np.sqrt((x1-x2)**2+(y1-y2)**2)<R):
                            flag = False
                            break

                if flag == True:
                    finalfeatures.append(features[Index])
                    s.add(Index)
                    if (len(s) > m):
                        break

            if(len(s)>m):
                break
            R-=10

        return  finalfeatures


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image,None)

## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''

#*****************Try2

        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage1=np.pad(grayImage,[(5,5),(5,5)],mode='constant')
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)
            x=x+5
            y=y+5

            each_desc=grayImage1[y-2:y+3,x-2:x+3].reshape([1,25])

            desc[i]=each_desc



            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            #raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        # newpd = np.zeros((height+4,width+4),dtype=float)
        # newpd[2:2+height,2:2+width] = grayImage
        #
        # desc = np.zeros((len(keypoints), 5 * 5))
        #
        # simple
        # for i, f in enumerate(keypoints):
        #     x,y = f.pt
        #     x, y = int(x), int(y)
        #     # TODO 4: The simple descriptor is a 5x5 window of intensities
        #     # sampled centered on the feature point. Store the descriptor
        #     # as a row-major vector. Treat pixels outside the image as zero.
        #
        #      desc[i] = np.reshape(newpd[2+x-2:2+x+3,2+y-2:2+y+3],(1,25))
        #      desc[i] = np.reshape(newpd[2 + y - 2:2 + y + 3, 2 + x - 2:2 + x + 3], (1, 25))

             # tempmat = np.zeros((5,5))
             # for row in range(-2, 3):
             #     for col in range(-2, 3):
             #         if 0<=y+row<grayImage.shape[0] and 0<=x+col<grayImage.shape[1]:
             #             tempmat[row + 2, col + 2] = grayImage[y+row, x+col]
             # tempmat = tempmat.reshape((1,25))
             # desc[i] = tempmat

            # TODO-BLOCK-BEGIN
            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END


        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8

        desc = np.zeros((len(keypoints), windowSize * windowSize))

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)
        height,width = grayImage.shape[:2]

        newpd = np.zeros((height+40-1,width+40-1),dtype=float)
        newpd[20:20+height,20:20+width] = grayImage

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            x,y = f.pt
            angle = -f.angle * (2*np.pi) / 360

            trans_mx1 = np.array([[1,0,-x],[0,1,-y],[0,0,1]])


            rot_mx = np.array([  [math.cos(angle), -math.sin(angle), 0],
                                 [math.sin(angle), math.cos(angle), 0],
                                 [0, 0, 1]])

            scale_mx = np.array([[1/5,0,0],
                                 [0,1/5,0],
                                 [0,0,1]])

            trans_mx2 = np.array([[1,0,4], [0,1,4], [0,0,1]])


            transMx = np.dot(trans_mx2,np.dot(scale_mx,np.dot(rot_mx,trans_mx1)))[0:2,0:3]



            # TODO-BLOCK-BEGIN

            
            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix

            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN

            destImage = destImage - np.mean(destImage)
            if(np.std(destImage)<10**-5):
                destImage = np.zeros((1,8*8))
            else:
                destImage = destImage / np.std(destImage)
                destImage = np.reshape(destImage, (1, 8 * 8))


            desc[i] = destImage
            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''

        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        orientationImage = np.zeros(grayImage.shape[:2],dtype=float)
        desc = np.zeros((len(keypoints),16*8))
        Ix = ndimage.sobel(grayImage,axis=1,mode='reflect')
        Iy = ndimage.sobel(grayImage,axis=0,mode='reflect')
        orientationImage  = np.rad2deg(np.arctan2(Iy,Ix))


        #SIFT
        for i,f in enumerate(keypoints):
            x,y = f.pt
            x = int(x)
            y = int(y)
            contain = np.zeros((16,8))
            for outrow in range(4):
                for outcol in range(4):
                    for inrow in range(4):
                        for incol in range(4):
                            distcol = outcol*4 + incol
                            distrow = outrow*4 + inrow

                            if(y-7+distrow)<0 or (y-7+distrow)>grayImage.shape[0]-1 or (x-7+distcol)<0 or (x-7+distcol)>grayImage.shape[1]-1:
                                break

                            degree = orientationImage[y-7+distrow,x-7+distcol]
                            if(degree<0):
                                degree+=360
                            degpart = int(degree//45)
                            contain[outrow*4+outcol,degpart] += 1

            contain = contain.reshape((1,128));
            stddev =np.std(contain)
            if stddev < 10**-5:
                contain = np.zeros((1,128))
            else:
                contain = (contain - np.mean(contain)) / stddev

            desc[i] = contain

        return desc
        # raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.

        # bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck = True)
        # matches = bf.match(desc1,desc2)


        # TODO-BLOCK-BEGIN



        # length = desc1.shape[1]
        # dist = distance.cdist(desc1,desc2,'euclidean')
        # for i in range(length):
        #     queryIdx = i
        #     imgIdx = np.argmin(dist[i,:])
        #     distan = dist[i,imgIdx]
        #     matches.append(cv2.DMatch(queryIdx,imgIdx,distan))

        # for i, desc in enumerate(desc1):
        #     dif = desc2 - desc
        #     sq = dif * dif
        #     sq = np.sum(sq,axis=1)
        #     bestInd = np.argmin(sq)
        #     match = cv2.DMatch()
        #     match.queryIdx = i
        #     match.trainIdx = bestInd
        #     match.distance = sq[bestInd]
        #     matches.append(match)

        n1 = desc1.shape[0]
        n2 = desc2.shape[0]
        distance = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')

        # print(distance)

        match = np.argmin(distance, 1)
        # print(match)
        for i in range(n1):
            f = cv2.DMatch()
            f.queryIdx = i
            f.trainIdx = int(match[i])
            f.distance = distance[i, int(match[i])]
            matches.append(f)


        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN

#********************Try1
        distance = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
        for col,row in enumerate(distance):
            trainIdx = col
            rowIdx = np.argsort(row)
            first= rowIdx[0]
            second = rowIdx[1]
            minest = row[first]
            minsec = row[second]
            ratio = minest / float(minsec)
            matches.append(cv2.DMatch(trainIdx, first, ratio))

#*********************Try2
        # length = desc1.shape[0]
        # dist = scipy.spatial.distance.cdist(desc1,desc2,'euclidean')
        # for i in range(length):
        #     m = cv2.DMatch()
        #     # m.queryIdx
        #     queryIdx= i
        #
        #     temp = dist[i]
        #     # m.imgIdx = np.argmin(temp)
        #     trainIdx = np.argmin(temp)
        #
        #     mindist = np.min(temp)
        #
        #     distbackup = np.delete(temp,trainIdx,axis=0)
        #     secmindist = distbackup[np.argmin(distbackup)]
        #     # m.distance = mindist / secmindist
        #     dista = mindist / secmindist
        #     matches.append(cv2.DMatch(queryIdx,trainIdx,dista))

#****************Try3
        # for i, desc in enumerate(desc1):
        #     dif = desc2 - desc
        #     sq = dif * dif
        #     sq = np.sum(sq,axis=1)
        #     bestInds = sq.argsort()[:2]
        #     match = cv2.DMatch()
        #     match.queryIdx = i
        #     match.trainIdx = bestInds[0]
        #     match.distance = sq[bestInds[0]] / sq[bestInds[1]]
        #     matches.append(match)

#*********************Try4
        # distance = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
        # width,height = distance.shape[:2]
        # newdistance = distance
        # minindex = np.argmin(distance,axis = 1)
        # min1 = np.min(distance,axis = 1)
        # #print(distance.shape[:2])
        # for i in range(width):
        #     newdistance[i,int(minindex[i])] = np.max(distance[i])
        # #newdistance = np.delete(distance, minindex,axis=1)
        # min2 = np.min(distance, axis = 1)
        # for i in range(desc1.shape[0]):
        #     f = cv2.DMatch()
        #     f.queryIdx = i
        #     f.trainIdx = int(minindex[i])
        #     f.distance = min1[i]/min2[i]
        #     #print(f.distance)
        #     matches.append(f)

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

