'''
Created on 30 Sep 2018

@author: pingshiyu
'''
'''
Given a path containing pickled image batches, make a pickled file of images of a specific size
'''
import glob
import pickle, numpy as np
import logging
import sys

# create logger
root = logging.getLogger()
logging.basicConfig(filename = '../logs/make_image_database.log',
                    level = logging.DEBUG,
                    filemode = 'w+',
                    format = '%(asctime)s %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
root.addHandler(ch)

def unpack_batch(batch):
    '''
    Batch packaged in list of (X, y) format. We unpack this into 2 numpy arrays Xs and ys
    Input:
        batch: list of tuples (X, y)
    Output:
        Xs, ys: numpy arrays, batch unzipped
    '''
    Xs_tuple, ys_tuple = zip(*batch)
    Xs, ys = np.array(Xs_tuple), np.array(ys_tuple)
    return Xs, ys

if __name__ == '__main__':
    image_path = '../images/database_square/female/*'
    max_num_images = 8000
    save_path = './data/female_faces_8000'
    
    total_batch_size = 0
    batch_paths = glob.iglob(image_path)
    image_batches, rating_age_batches = [], []
    for batch_path in batch_paths:
        logging.info('reading file %s' % batch_path)
        # unpacking batch
        with open(batch_path, 'rb') as f:
            image_batch, rating_age_batch = unpack_batch(pickle.load(f))
            
        # checking we are within size limit
        batch_size = np.ma.size(image_batch, 0)
        logging.info('current batch size %s' % batch_size)
        if (total_batch_size + batch_size >= max_num_images):
            # current batch brings us over the max image - take whats needed from this batch
            remaining_num = max_num_images - total_batch_size
            image_batches.append(image_batch[:remaining_num])
            rating_age_batches.append(rating_age_batch[:remaining_num])
            break
        else:
            # current batch is saved totally
            image_batches.append(image_batch)
            rating_age_batches.append(rating_age_batch)
            total_batch_size += batch_size
        logging.info('total images read so far %s' % total_batch_size)
            
    # joining the batches together
    all_images = np.concatenate(image_batches, axis=0)
    all_rating_age = np.concatenate(rating_age_batches, axis=0)
    # saving the big batch
    with open(save_path, 'wb') as f:
        pickle.dump((all_images, all_rating_age), f)
        logging.info('%s files saved in %s' % (max_num_images, save_path))
        
    