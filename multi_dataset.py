import numpy as np
from multiprocessing import Pool
import os

# transform transitions to training data by multiprocess

def read_data(batch_num):
    print str(batch_num)+" start!"
    dataset=np.zeros([1,256,256,3], dtype=np.uint8) # temp
    for i in range(256):
        tran_file_path = \
                    '/home/baxter/catkin_ws/src/huang/scripts/explore_transitions/exp_transition%d.npy' \
                    % (i+ 256*batch_num)
        transition = np.load(tran_file_path, allow_pickle=True)
        transition = transition.tolist()
        data = np.concatenate((np.array(transition['observe0_img']),np.array(transition['observe1_img'])),axis=0)
        dataset = np.concatenate((dataset, data), axis=0)
    dataset = np.delete(dataset, 0, 0) # delete temp row
    file_path = \
        '/home/baxter/Documents/beta-vae/dataset/dataset%d.npy'% (batch_num)
    np.save(file_path, dataset, allow_pickle=True)


if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(12)
    for k in range(82):
        p.apply_async(read_data, args=(k,))
    p.close()
    p.join()
    print('All subprocesses done.')
