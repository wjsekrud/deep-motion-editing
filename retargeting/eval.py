import os
from models import create_model
from datasets import create_dataset, get_character_names
import option_parser
import torch
from tqdm import tqdm


def eval(eval_seq, save_dir, characters, epoch=1000, test_device='cpu'):
    para_path = os.path.join(save_dir, 'para.txt') 
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]#저장 경로에 존재하는 para.txt로부터 명령줄 인수를 불러온다
        args = option_parser.get_parser().parse_args(argv_) #train 당시의 모델과 같은 사양의 모델을 불러와야 하기 때운
        
    # 아래 과정에서 테스트 환경 구성을 위해 몇 가지 인수들을 바꿔준다
    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq
    
    character_names = characters
    tmp = character_names[1][args.eval_seq]
    character_names[1][args.eval_seq] = character_names[1][0]
    character_names[1][0] = tmp

    args.save_dir = save_dir
    character_names = characters
    print(f"cnames: {character_names}")

    dataset = create_dataset(args, character_names)

    model = create_model(args, character_names, dataset)
    model.load(epoch) #특정 epoch의 모델을 불러온다

    for i, motions in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(motions) #데이터셋의 각 모션들을 의미함
        model.test() #만들어진 dataset으로 모델을 테스트한다. foward->compute_test_result의 과정을 거친다.
        # 이 테스트는 1개의 애니메이션 당 1 번 수행된다
        #print(f"DoneTestIter{i}")

if __name__ == '__main__':
    parser = option_parser.get_parser()
    args = parser.parse_args()
    eval(args.eval_seq, args.save_dir, args.cuda_device)
