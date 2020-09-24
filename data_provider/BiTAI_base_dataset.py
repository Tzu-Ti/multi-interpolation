import os
import random
import re
import traceback
from warnings import warn

import cv2
import imageio
import numpy as np

class InputHandle:
    def __init__(self, input_param):
        """Constructor

        :param c_dim: The number of color channels each output video should have
        :param video_list_path: The path to the video list text file
        :param K: The number of preceding frames
        :param T: The number of future or middle frames
        :param backwards: Flag to allow data augmentation by randomly reversing videos temporally
        :param flip: Flag to allow data augmentation by randomly flipping videos horizontally
        :param image_size: The spatial resolution of the video (W x H)
        :param F: The number of following frames
        """
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.flip = True
        self.backwards = True
        self.sameBatchMode = False
        self.c_dim = input_param['img_size'][-1]
        self.data.update({'dims':input_param['img_size']}) #[h,w,c]
        # if 'KTH' in str(np.squeeze(input_param['paths'])):
        #     image_size = [128,128]
        #     self.data.update({'dims':[128,128,3]})
        # elif 'nctu' in str(np.squeeze(input_param['paths'])):
        #     image_size = [128,220]
        #     self.data.update({'dims':[128,220,3]})
        # else:
        #     image_size = [128,128]
        #     self.data.update({'dims':[128,128,3]})
        
        if 'test' in str(np.squeeze(input_param['paths'])):
            self.sameBatchMode = True
            self.flip = False
            self.backwards = False

        
        self.image_size = input_param['img_size'][0:-1] #[h,w]
        self.resample_on_fail = True

        self.seq_len = input_param['seq_len']

        self.vid_path = None
        self.load()

    def load(self):
        video_list_path = self.paths[0]
        # Read the list of files
        with open(video_list_path, 'r') as f:
            self.files = [line.strip() for line in f.readlines()]

    def total(self):
        #print(len(self.files))
        return len(self.files)

    def begin(self, do_shuffle = False):
        self.indices = np.arange(self.total(),dtype="int32")
        #print('self.indices= ', self.indices)
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.sameBatchMode == True:
            if self.current_position + 1 <= self.total():
                self.current_batch_size = self.minibatch_size
            else:
                self.current_batch_size = self.total() - self.current_position
            self.current_batch_indices = [self.indices[self.current_position] for i in range(self.current_batch_size)]
        else:
            if self.current_position + self.minibatch_size <= self.total():
                self.current_batch_size = self.minibatch_size
            else:
                self.current_batch_size = self.total() - self.current_position
            self.current_batch_indices = self.indices[
                self.current_position:self.current_position + self.current_batch_size]
            

    def next(self):
        if self.sameBatchMode == True:
            self.current_position += 1
            if self.no_batch_left():
                return None
            if self.current_position + 1 <= self.total():
                self.current_batch_size = self.minibatch_size
            else:
                self.current_batch_size = self.total() - self.current_position
            self.current_batch_indices = [self.indices[self.current_position] for i in range(self.current_batch_size)]
        else:
            self.current_position += self.current_batch_size
            if self.no_batch_left():
                return None
            if self.current_position + self.minibatch_size <= self.total():
                self.current_batch_size = self.minibatch_size
            else:
                self.current_batch_size = self.total() - self.current_position
            self.current_batch_indices = self.indices[
                self.current_position:self.current_position + self.current_batch_size]
        #print('current_batch_indices= ', self.current_batch_indices)
        #current_input_length = 1/2 len of seq forever when begin/range = 1 [input/output, index, begin/range]
        self.current_input_length = 10
        #current_output_length = 1/2 len of seq forever when begin/range = 1 [input/output, index, begin/range]
        self.current_output_length = 10

    def no_batch_left(self):
        if self.sameBatchMode == True:
            if self.current_position > self.total() - 1:
                return True
            else:
                return False
        else:
            if self.current_position > self.total() - self.current_batch_size:
                return True
            else:
                return False

    def input_batch(self, isRev):
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.current_batch_size, self.current_input_length) +
            tuple(self.data['dims'][0])).astype(self.input_data_type)
        input_batch = np.transpose(input_batch,(0,1,3,4,2))
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            if isRev == False:
                begin = self.data['clips'][0, batch_ind, 0]
                end = self.data['clips'][0, batch_ind, 0] + \
                        self.data['clips'][0, batch_ind, 1]
            else:
                begin = self.data['clips'][0, batch_ind, 0]
                end = self.data['clips'][0, batch_ind, 0] + \
                        self.data['clips'][0, batch_ind, 1]
            print('input:   begin:', begin,'end:', end)
            data_slice = self.data['input_raw_data'][begin:end, :, :, :]
            data_slice = np.transpose(data_slice,(0,2,3,1))
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def read_seq(self, vid, frame_indexes, clip_label):
        """Obtain a video clip along with corresponding difference frames and auxiliary information.

        Returns a dict with the following key-value pairs:
        - vid_name: A string identifying the video that the clip was extracted from
        - start-end: The start and end indexes of the video frames that were extracted (inclusive)
        - targets: The full video clip [C x H x W x T FloatTensor]
        - diff_in: The difference frames of the preceding frames in the full video clip [C x H x W x T_P]
        - diff_in_F: The difference frames of the following frames in the full video clip [C x H x W x T_F]

        :param vid: An imageio Reader
        :param stidx: The index of the first video frame to extract clips from
        :param vid_name: A string identifying the video that the clip was extracted from
        :param vid_path: The path to the given video file
        """

        targets = []

        # generate [0, 1] random variable to determine flip and backward
        flip_flag = self.flip and (random.random() > 0.5)
        back_flag = self.backwards and (random.random() > 0.5)

        # read and process in each frame
        for clip_index, t in enumerate(frame_indexes):
            # read in one frame from the video
            vid_frame = self.get_frame(vid, t)
            if vid_frame is None:
                warn('Failed to read the given sequence of frames (frame %d in %s)' % (t, vid._filename))
                return None

            # resize frame
            # # # img = cv2.resize(vid_frame, (self.image_size[1], self.image_size[0]))[:, :, ::-1]
            img = cv2.resize(vid_frame, (self.image_size[1], self.image_size[0]))
            
#             # replace second, fourth, eighth, tenth frame with random noise
#             if (clip_index == 1 or clip_index == 3 or clip_index == 7 or clip_index == 9):
#                 noise = np.random.normal(0, 255, img.shape)
#                 img = noise
            
            img = img / 255.0
            # flip the input frame horizontally
            if flip_flag:
                img = img[:, ::-1, :]
            
            if self.c_dim == 1:
                img = self.bgr2gray(img)
                
            # pad the image
            # # img = cv2.copyMakeBorder(img, 0, self.padding_size[0], 0, self.padding_size[1], cv2.BORDER_CONSTANT, -1)

            targets.append(img)

        # Reverse the temporal ordering of frames
        if back_flag:
            targets = targets[::-1]

        

        # stack frames and map [0, 1] to [-1, 1]
        # # target = fore_transform(torch.stack(targets))  # T x C x H x W
        # if number of color channels is 1, use the gray scale image as input
        # # if self.c_dim == 1:
        # #     target = bgr2gray(target)

        # # ret = {
        # #     'targets': target,
        # #     'clip_label': clip_label
        # # }

        return targets

    def bgr2gray(self, image):
        # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
        gray_ = 0.1140 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.2989 * image[:, :, 2]
        gray = np.expand_dims(gray_, 2)
        return gray

    def open_video(self, vid_path):
        """Obtain a file reader for the video at the given path.

        Wraps the line to obtain the reader in a while loop. This is necessary because it fails randomly even for
        readable videos.

        :param vid_path: The path to the video file
        """
        for _ in xrange(5):
            try:
                vid = imageio.get_reader(vid_path, 'ffmpeg')
                return vid
            except IOError:
                traceback.print_exc()
                warn('imageio failed in loading video %s, retrying' % vid_path)

        warn('Failed to load video %s after multiple attempts, returning' % vid_path)
        return None


    def get_frame(self, vid, frame_index):
        for _ in xrange(5):
            try:
                frame = vid.get_data(frame_index)
                return frame
            except imageio.core.CannotReadFrameError:
                traceback.print_exc()
                warn('Failed to read frame %d in %s, retrying' % (frame_index, vid._filename))

        warn('Failed to read frame %d in %s after five attempts. Check for corruption' % (frame_index, vid._filename))
        return None
    
    def get_batch(self, isRev):
        """Obtain data associated with a video clip from the dataset (BGR video frames)."""

        # Try to use the given video to extract clips
        input_batch = np.zeros(
            (self.current_batch_size, self.seq_len) +
            tuple(self.data['dims'])).astype(self.input_data_type)
        i = 0
        while i < self.current_batch_size:
            batch_ind = self.current_batch_indices[i]
            #print('self.files[batch_ind]= ', self.files[batch_ind])

            # Parse the line for the given index
            split_line = self.files[batch_ind].split()
            if len(split_line) == 1:
                video_file_path = split_line[0]
            else:
                video_file_path, full_range_str = split_line
                #print('full_range_str= ', full_range_str)

            # Open the video
            vid = self.open_video(video_file_path)
            if vid is None:
                if not self.resample_on_fail:
                    raise RuntimeError('Video at %s could not be opened' % video_file_path)
                # Video could not be opened, so try another video
                self.current_batch_indices[i] = np.random.randint(0, len(self.files))
                print('vid is None', 'i=',i, 'file=', video_file_path.split('/')[-1])
                continue

            # Use the whole video or, if specified, only use the provided 1-indexed frame indexes
            # Note: full_range is a 0-indexed, inclusive range
            if len(split_line) == 1:
                full_range = (0, vid.get_length()-1)
                #print('full_range= ', full_range)
            else:
                # Convert to 0-indexed indexes
                full_range = list(int(d)-1 for d in full_range_str.split('-'))
                #print('full_range_str= ', full_range_str, 'full_range= ', full_range)
            full_range_length = full_range[1] - full_range[0] + 1
            if full_range_length < self.seq_len and self.seq_len > 20:
                full_range[1] = full_range[1] + 10
            elif full_range_length < self.seq_len:
                if not self.resample_on_fail:
                    raise RuntimeError('Interval %s in video %s is too short' % (str(full_range), video_file_path))
                # The video clip length is too short, so try another video
                self.current_batch_indices[i] = np.random.randint(0, len(self.files))
                print('full_range_length < self.seq_len', 'i=',i, 'file=', video_file_path.split('/')[-1])
                continue

            # Randomly choose a sub-range (inclusive) within the full range
            # Note: random.randint(a, b) chooses within closed interval [a, b]
            start_index = random.randint(full_range[0], full_range[1] - self.seq_len + 1)
            frame_indexes = range(start_index, start_index + self.seq_len)
            # Select the chosen frames
            try:
                clip_label = '%s_%d-%d' % (os.path.basename(video_file_path), full_range[0] + 1, full_range[1] + 1)
                item = self.read_seq(vid, frame_indexes, clip_label)
            except IndexError as e:
                warn('IndexError for video at %s, video length %d, video range %s, indexes %s' \
                        % (video_file_path, vid.get_length(), full_range_str, str(frame_indexes)))
                if "Reached end of video" in str(e):
                    #print('Reached end of video')
                    self.current_batch_indices[i] = np.random.randint(0, len(self.files))
                    continue
                else:
                    raise e
            if item is None:
                if not self.resample_on_fail:
                    raise RuntimeError('Failed to sample frames starting at %d in %s' % (start_index, video_file_path))
                # Failed to load the given set of frames, so try another item
                self.current_batch_indices[i] = np.random.randint(0, len(self.files))
                print('item is None', 'i=',i, 'start_end=',start_index, '-', start_index + self.seq_len)
                continue
            
            data_slice = item #raw data frames
            input_batch[i, :self.seq_len, :, :, :] = data_slice
            print('file=', video_file_path.split('/')[-1], ' begin:', start_index + 1,'end:', start_index + self.seq_len)
            i = i + 1

        fileName = str(video_file_path.split('/')[-1])+'_'+str(start_index + 1)+'-'+str(start_index + self.seq_len)
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch, fileName
