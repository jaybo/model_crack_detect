import glob
import numpy as np
import cv2
import random
import imgaug as ia
from imgaug import augmenters as iaa

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
        if self.label_dir:
            c = list(zip(self.images, self.masks))
            random.shuffle(c)
            self.images, self.masks = zip(*c)
        else:
            random.shuffle(self.images)

        # make training and validation sets
        training_set = np.random.choice([True, False], self.image_count, p=[self.training_split, 1.0-self.training_split])
        not_training_set = np.invert(training_set)
        self.training_images = [x for x,y in zip(self.images, training_set) if y]
        self.validation_images = [x for x,y in zip(self.images, not_training_set) if y]
        if self.label_dir:
            self.training_masks = [x for x,y in zip(self.masks, training_set) if y]
            self.validation_masks = [x for x,y in zip(self.masks, not_training_set) if y]
        self.batch_size = batch_size
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

        sometimes = lambda aug: iaa.Sometimes(0.5, aug) # 50% of the time

        self.seq = iaa.Sequential([
            #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Flipud(0.5),
            sometimes(iaa.Affine(
                 #scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                 translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                 rotate=(-35, 35)
                 )),
            #iaa.GaussianBlur((0, 1.0))
            #iaa.AddToHueAndSaturation((-10, 10), name="AddToHueAndSaturation"),
            iaa.Multiply((0.95, 1.05), name="Multiply")
            # iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        ])

        def activator_mask(images, augmenter, parents, default):
            if augmenter.name in ["AddToHueAndSaturation", "Multiply", "GausianBlur"]:
                return False
            else:
                # default value for all other augmenters
                return default
        self.hooks_activator_mask = ia.HooksImages(activator=activator_mask)

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
            percentage = 97 # ignore (1-percentage) top bright values
            percentage_target = 180 # target mean gray level
            a = np.percentile(img, percentage)
            if a < percentage_target:
                img = cv2.multiply(img, percentage_target / a)
                # print (a, np.percentile(img, percentage))

            img = cv2.equalizeHist(img)
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
                            if tmsk.any() > 0:
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

        return img, mask


    def training_batch(self):
        while True:
            image_list = []
            mask_list = []

            for i in range(self.batch_size):
                img, mask = self.get_image_and_mask(self.epoch_index, source='training', augment=False)

                self.epoch_index += 1
                if self.epoch_index >= self.steps_per_epoch:
                    self.epoch_index = 0
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
        while True:
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
            seq_det = self.seq.to_deterministic()
            image_list = seq_det.augment_images(image_list)
            mask_list = seq_det.augment_images(mask_list, hooks=self.hooks_activator_mask)

            image_list = np.array(image_list, dtype=np.float32) #Note: don't scale input, because use batchnorm after input

            # image_list /= 255.

            image_list = image_list.reshape(self.validation_batch_size, self.height, self.width, 1)
            mask_list = np.array(mask_list, dtype=np.float32)
            mask_list /= 255.0 # [0,1]

            mask_list= mask_list.reshape(self.validation_batch_size,self.height*self.width, 1) #NUMBER_OF_CLASSES

            yield image_list, mask_list

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
                ret,thresh = cv2.threshold(mask8,127,255,0)
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
