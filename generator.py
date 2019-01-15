import glob
import numpy as np
import cv2
import random
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(1)

MEAN = 187.41119924316425 
VARIANCE = 7.836514274501137

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def contrast_local_normalization(img):

    float_gray = img.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur

    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)

    gray = num / den

    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    gray = gray * 255
    gray = gray.astype(np.uint8)
    return gray


class batch_generator(object):
    ''' generator for images and masks '''
    def __init__(self,
             image_dir=None,
             label_dir=None,
             batch_size=None,
             validation_batch_size=None,
             training_split=0.8,
             output_size=None,
             scale_size=None,
             include_only_crops_with_mask=False,
             augment=True):
        '''
        output_size is a tuple (H, W) and is the final output size. 
        scale_size is a tuple (H, W) and only used when cropping.
        original_image -> scale_size -> crop to output_size.
        include_only_crops_with_mask means skip crops which don't have at least one mask bit set
        '''

        self.training_split = training_split
        self.output_size = output_size
        self.scale_size = scale_size
        self.include_only_crops_with_mask = include_only_crops_with_mask
        self.augment=augment

        # get lists of img and mask
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(glob.glob(image_dir))
        # filter them
        exclude_prefix = ['BrightField', 'DarkField', '_montage']
        self.images = [x for x in self.images if not any(stop in x for stop in exclude_prefix) ]

        self.image_count = len(self.images)
        if self.label_dir:
            self.masks = sorted(glob.glob(label_dir))
            self.mask_count = len(self.masks)
            assert (self.image_count == self.mask_count)

        # shuffle
        self.shuffle()

        # make training and validation sets
        training_set = np.random.choice([True, False], self.image_count, p=[self.training_split, 1.0-self.training_split])
        not_training_set = np.invert(training_set)
        self.training_images = [x for x,y in zip(self.images, training_set) if y]
        self.validation_images = [x for x,y in zip(self.images, not_training_set) if y]
        if self.label_dir:
            self.training_masks = [x for x,y in zip(self.masks, training_set) if y]
            self.validation_masks = [x for x,y in zip(self.masks, not_training_set) if y]
        self.batch_size = batch_size
        if validation_batch_size is None:
            validation_batch_size = len(self.validation_images)
        self.validation_batch_size = validation_batch_size
        self.steps_per_epoch = int(len(self.training_images) / batch_size)
        self.validation_steps = int(len(self.validation_images) / validation_batch_size)
        self.epoch_index = 0
        self.validation_index = 0
        #self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # figure out the size of each image
        img = cv2.imread(self.images[0], cv2.IMREAD_COLOR)
        if self.output_size:
            self.height = self.output_size[0]
            self.width = self.output_size[1]
        else:
            self.height = img.shape[0]
            self.width = img.shape[1]

        if self.scale_size and self.output_size and self.output_size < self.scale_size:
            self.cropping = True
        else:
            self.cropping = False

        sometimes7 = lambda aug: iaa.Sometimes(0.75, aug) # % of the time
        sometimes3 = lambda aug: iaa.Sometimes(0.33, aug) # % of the time

        self.seq = iaa.Sequential([
            #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Flipud(0.5),
            sometimes7(iaa.Affine(
                 #scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                 translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                 rotate=(-45, 45),
                 mode='constant',
                 cval= MEAN
                 )),
            #iaa.GaussianBlur((0, 1.0))
            #iaa.AddToHueAndSaturation((-10, 10), name="AddToHueAndSaturation"),
            sometimes3(iaa.Multiply((0.95, 1.05), name="Multiply"))
            # iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        ])

        def activator_mask(images, augmenter, parents, default):
            if augmenter.name in ["AddToHueAndSaturation", "Multiply", "GausianBlur"]:
                return False
            else:
                # default value for all other augmenters
                return default
        self.hooks_activator_mask = ia.HooksImages(activator=activator_mask)
        
        self.template = cv2.imread(r"../data/crack_detect/original/crack_images/20170505160314874_295434_5LC_0064_01_001088_0_14_74.tif", cv2.IMREAD_GRAYSCALE)


    def shuffle (self):
        ''' randomize the images and masks (if available) '''
        if self.label_dir:
            c = list(zip(self.images, self.masks))
            random.shuffle(c)
            self.images, self.masks = zip(*c)
        else:
            random.shuffle(self.images)


    def get_image_and_mask(self, index, source='training', augment=False):
        ''' return the optionally scaled image and mask from one of the 3 sets'''
        if source == 'training':
            img_src = self.training_images
            if self.label_dir:
                mask_src = self.training_masks
        elif source == 'validation':
            img_src = self.validation_images
            if self.label_dir:
                mask_src = self.validation_masks
        elif source == 'all':
            img_src = self.images
            if self.label_dir:
                mask_src = self.masks

        img = cv2.imread(img_src[index], cv2.IMREAD_GRAYSCALE)
        mask = None
        if self.label_dir:
            mask = cv2.imread(mask_src[index], cv2.IMREAD_GRAYSCALE)

        if True:
            # make really, really, really dark images lighter
            # percentage = 97 # ignore (1-percentage) top bright values
            # percentage_target = 180 # target mean gray level
            # a = np.percentile(img, percentage)
            # if a < percentage_target:
            #     img = cv2.multiply(img, percentage_target / a)
            #     # print (a, np.percentile(img, percentage))

            # img = contrast_local_normalization(img)
            
            #img -= 187 # 187.4111992431641
            # img /= 7.840043441207606

            #img = cv2.equalizeHist(img)
            
            #img = img / 255.0

            #img = hist_match(img, self.template)

            # just subtract the mean
            # mean = np.mean(img)
            # img -= np.uint8(mean)

            pass
            
        if self.output_size:
            # somehow changing the output size
            if self.cropping:
                # first scale, then crop
                timg = tmsk = None
                # resize
                img = cv2.resize(img, (self.scale_size[1], self.scale_size[0]))
                if mask is not None:
                    mask = cv2.resize(mask, (self.scale_size[1], self.scale_size[0]))
                for j in range(10):
                    # crop
                    y = random.randint(0, self.scale_size[0] - self.output_size[0])
                    x = random.randint(0, self.scale_size[1] - self.output_size[1])
                    timg = img[y:y+self.output_size[0], x:x+self.output_size[1]]
                    if mask is not None:
                        tmsk = mask[y:y+self.output_size[0], x:x+self.output_size[1]]
                        if self.include_only_crops_with_mask:
                            if np.count_nonzero(tmsk) > 5000:
                                break
                if timg is not None:
                    img = timg.copy()
                if tmsk is not None:
                    mask = tmsk.copy()

            else:
                # return original image size
                img = cv2.resize(img, (self.output_size[1], self.output_size[0]))
                if mask is not None:
                    mask = cv2.resize(mask, (self.output_size[1], self.output_size[0]))

        if mask is not None:
            mask *= 255

        if augment:
            seq_det = self.seq.to_deterministic()
            img = seq_det.augment_image(img)
            if self.label_dir:
                mask = seq_det.augment_image(mask, hooks=self.hooks_activator_mask)
                mask[mask == int(MEAN)] = 0

        # mean and stddev correction
        # mean = 187.4111992431641 
        # stddev = 7.840043441207606 
        # cv2.subtract(img, mean, img, mask=None, dtype='uint8')
        # cv2.divide(img, stddev, img, mask=None, dtype='uint8')
        # img -=187
        #img = img / 8

        return img, mask


    def training_batch(self):
        while True:
            image_list = []
            mask_list = []

            for i in range(self.batch_size):
                img, mask = self.get_image_and_mask(self.epoch_index, source='training', augment=True)

                self.epoch_index += 1
                if self.epoch_index >= self.steps_per_epoch:
                    self.epoch_index = 0
                    self.shuffle()
                image_list.append(img)
                mask_list.append(mask)

            seq_det = self.seq.to_deterministic()
            image_list = seq_det.augment_images(image_list)
            mask_list = seq_det.augment_images(mask_list, hooks=self.hooks_activator_mask)

            image_list = np.array(image_list, dtype=np.float32) #Note: don't scale input, because use batchnorm after input
            
            # image_list /= 255.
            
            image_list = image_list.reshape(self.batch_size, self.height, self.width, 1)
            mask_list = np.array(mask_list, dtype=np.float32)
            
            mask_list /= 255.0 # [0,1]

            mask_list= mask_list.reshape(self.batch_size,self.height*self.width, 1) #NUMBER_OF_CLASSES

            yield image_list, mask_list

    def validation_batch(self):
        ''' if batch_size == None, then batch_size == epoch '''
        
        image_list = []
        mask_list = []

        for i in range(self.validation_batch_size):
            img, mask = self.get_image_and_mask(self.validation_index, source='validation', augment=False)

            self.validation_index += 1
            if self.validation_index >= self.validation_steps:
                self.validation_index = 0
            image_list.append(img)
            mask_list.append(mask)

        # don't augment the validation batch
        # seq_det = self.seq.to_deterministic()
        # image_list = seq_det.augment_images(image_list)
        # mask_list = seq_det.augment_images(mask_list, hooks=self.hooks_activator_mask)

        image_list = np.array(image_list, dtype=np.float32) #Note: don't scale input, because use batchnorm after input

        # image_list /= 255.

        image_list = image_list.reshape(self.validation_batch_size, self.height, self.width, 1)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0 # [0,1]

        mask_list= mask_list.reshape(self.validation_batch_size,self.height*self.width, 1) #NUMBER_OF_CLASSES

        return image_list, mask_list

    def get_random_validation(self):
        index = randint(0, self.validation_steps)
        return self.get_image_and_mask(index, source='validation')


    def show_all(self):
        ''' overlay the mask onto the image '''
        for index in range(self.image_count):
            img, mask8 = self.get_image_and_mask(index, source='all', augment=True)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            if mask8 is not None:
                mask = np.zeros_like(img)
                mask[:,:,2] = mask8
                alpha = 0.25
                cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)
                # outline contour
                ret,thresh = cv2.threshold(mask8,254,255,0)
                im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0,255,0), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            #img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
            font_scale=0.5
            cv2.putText(img,self.images[index],(20,20), font, font_scale,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(img,self.images[index],(22,22), font, font_scale,(255,255,255),1,cv2.LINE_AA)
            
            cv2.imshow('img',img)
            key = cv2.waitKey(0)
            if key == 27: # esc
                break
        cv2.destroyAllWindows()


def calc_target_bright_value(image_dir, percentage=97):
    ''' Find the bright percentage value for each image '''
    import matplotlib.pyplot as plt

    bg = batch_generator(
        image_dir,
        None,
        batch_size=1,
        validation_batch_size=1,
        output_size=(1920, 1920), # (H, W)
        scale_size=None, # (H, W)
        include_only_crops_with_mask = False,
        augment=False)

    p = []
    for index in range(bg.image_count):
        img, mask8 = bg.get_image_and_mask(index, source='all', augment=False)
        a = np.percentile(img, percentage)
        if a > 250:
            continue # ignore huge white patches
        p.append (a)

    print (np.min(p), np.max(p), np.mean(p))
    plt.hist(p, bins=50)
    plt.show()

if __name__ == '__main__':

    # calc_target_bright_value(r"..\data\crack_detect\original\crack_images\*.tif")

    bg = batch_generator(
        image_dir=r"..\data\crack_detect\original\crack_images\*.tif",
        label_dir=r"..\data\crack_detect\original\crack_annotations\*.png",
        batch_size=1,
        validation_batch_size=1,
        output_size=(480, 480), # (H, W)
        scale_size=(1920, 1920), # (H, W)
        include_only_crops_with_mask = True
        ) 
    #img, mask = bg.training_batch(batch_size=8)
    bg.show_all()
    #im, mask = bg.get_random_validation()
