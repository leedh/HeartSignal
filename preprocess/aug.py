import random
import numpy as np


def new_augment(self, scale=1):        
        # data augmentation 코드 작성해야한다.

        # augment normal
        aug_nos = scale*len(self.classwise_cycle_list[0]) - len(self.classwise_cycle_list[0])
        for idx in range(aug_nos):
            # normal_i + normal_j
            i = random.randint(0, len(self.classwise_cycle_list[0])-1)
            j = random.randint(0, len(self.classwise_cycle_list[0])-1)
            normal_i = self.classwise_cycle_list[0][i]
            normal_j = self.classwise_cycle_list[0][j]
            new_sample = np.concatenate([normal_i[0], normal_j[0]])
            self.cycle_list.append((new_sample, 0, normal_i[2]+'-'+normal_j[2],
                idx, 0))
            self.filenames_with_labels.append(normal_i[2]+'-'+normal_j[2]+'_'+str(idx)+'_0')