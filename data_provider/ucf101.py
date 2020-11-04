import random
import traceback
from warnings import warn

import numpy as np
import imageio

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.data.update({'dims':input_param['img_size']})
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.seq_len = input_param['seq_len']
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
        self.indices = np.arange(self.total(), dtype="int32")
        #print('self.indices= ', self.indices)
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        
    def no_batch_left(self):
        if self.current_position >= self.total() - self.current_batch_size:
            return True
        else:
            return False
        
    def open_video(self, vid_path):
        for _ in xrange(5):
            try:
                vid = imageio.get_reader(vid_path, 'ffmpeg')
                return vid
            except IOError:
                traceback.print_exc()
                warn('imageio failed in loading video %s, retrying' % vid_path)
            
            warn('Faild to load video %s after multiple attempts, returning' % vid_path)
            return None
        
    def get_batch(self, isRev):
        input_batch = np.zeros(
            (self.current_batch_size, self.seq_len) +
            tuple(self.data['dims'])).astype(self.input_data_type) # (minibatch, seq_length, H, W, C)
        
        i = 0
        while i < self.current_batch_size:
            batch_ind = self.current_batch_indices[i]
            
            # Parse the line for the given index
            split_line = self.files[batch_ind].split()
            if len(split_line) == 1:
                video_file_path = split_line[0]
            else:
                video_file_path, full_range_str = split_line
                
            # Open the video
            vid = self.open_video(video_file_path)
            if vid is None:
                self.current_batch_indices[i] = np.random.randint(0, len(self.files))
                print('vid is None', 'i=',i, 'file=', video_file_path.split('/')[-1])
                continue
            print('vid: {}'.format(vid))
            import sys
            sys.exit()