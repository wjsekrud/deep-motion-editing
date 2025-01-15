import os
from os.path import join as pjoin
from get_error import full_batch
import numpy as np
from option_parser import try_mkdir
from eval import eval
import argparse
from datasets import get_character_names
import random 

def batch_copy(source_path, suffix, dest_path, dest_suffix=None):
    try_mkdir(dest_path)
    files = [f for f in os.listdir(source_path) if f.endswith('_{}.bvh'.format(suffix))]

    length = len('_{}.bvh'.format(suffix))
    for f in files:
        if dest_suffix is not None:
            cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '_{}.bvh'.format(dest_suffix)))
        else:
            cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '.bvh'))
        os.system(cmd)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser() #이번엔 사전에 정의된 arg는 따로 없다
    parser.add_argument('--save_dir', type=str, default='./pretrained/') #저장 경로가 정의되어 있지 않다면 pretrain에 저장한다
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    prefix = args.save_dir
    
    cross_dest_path = pjoin(prefix, 'results/cross_structure/') #서로 다른 스켈레톤 간의 리타깃 결과
    intra_dest_path = pjoin(prefix, 'results/intra_structure/') #같은 스켈레톤(개수) 간의 리타깃 결과
    source_path = pjoin(prefix, 'results/bvh/') #평가 대상이 되는 bvh들의 경로이다. eval을 할 떄 생성됨
    args.is_train = True
    

    characters = get_character_names(args)
    char_B = random.sample(characters[1], min(min(len(characters[0]), len(characters[1])), 4))
    characters[1] = char_B
    flat_characters = sum(characters, [])

    
    cross_error = []
    intra_error = []
    for i in range(len(characters)):
        print('Batch [{}/4]'.format(i + 1))
        eval(i, prefix, characters, args.epochs) #reatargeting/eval.py에서 불러와 직접적으로 평가를 하는 부분, batch마다 한 번 호출된다. i 는 eval_seq
        
        print('Collecting test error...')
        if i == 0:
            cross_error += full_batch(0, prefix, flat_characters)
            for char in flat_characters:
                print(char)
                batch_copy(os.path.join(source_path, char), 0, os.path.join(cross_dest_path, char))
                batch_copy(os.path.join(source_path, char), 'gt', os.path.join(cross_dest_path, char), 'gt')

        intra_dest = os.path.join(intra_dest_path, 'from_{}'.format(flat_characters[i]))
        for char in flat_characters:
            print(f"Copying batch for {char}...")
            for char in flat_characters:
                batch_copy(os.path.join(source_path, char), 1, os.path.join(intra_dest, char))
                batch_copy(os.path.join(source_path, char), 'gt', os.path.join(intra_dest, char), 'gt')
            

        intra_error += full_batch(1, prefix, flat_characters)

    cross_error = np.array(cross_error)
    intra_error = np.array(intra_error)

    cross_error_mean = cross_error.mean()
    intra_error_mean = intra_error.mean()

    os.system('rm -r %s' % pjoin(prefix, 'results/bvh'))

    print('Intra-retargeting error:', intra_error_mean)
    print('Cross-retargeting error:', cross_error_mean)
    print('Evaluation finished!')
