import sys
sys.path.append('./retargeting/')
from torch.utils.data.dataloader import DataLoader
from models import create_model
from datasets import create_dataset, get_train_A_characters, get_train_B_characters
import option_parser
import os
from option_parser import try_mkdir
import time


def main():
    args = option_parser.get_args() #명령줄 인수로 받은 옵션들을 파싱한다
    
    characters = [get_train_A_characters(), get_train_B_characters()]
    print(characters)
                  
    #또한 이 characters의 길이 자체는 A와 B의 2이고, 각 A와 B를 이루는 리스트에는 서로 다른 캐릭터들이 들어갈 수 있다

    log_path = os.path.join(args.save_dir, 'logs/')
    try_mkdir(args.save_dir) #학습 결과를 저장할 폴더를 찾고 없으면 만든다
    try_mkdir(log_path)

    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file: #방금 받은 명령줄 인수를 텍스트 파일로서 저장 경로에 쓴다
        para_file.write(' '.join(sys.argv))

    dataset = create_dataset(args, characters) #datasets/__init__.py 로부터 create_dataset을 호출, 이 떄 is_train = 1이므로 combined_motion.py의 MixedData 클래스가 생성된다.
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = create_model(args, characters, dataset) # create_model 은 models/architecture.py에서 GAN_Model 클래스를 생성한다.

    if args.epoch_begin:
        model.load(epoch=args.epoch_begin, download=False)

    model.setup() #부모 클래스의 함수 / 스케줄러를 초기화한다

    start_time = time.time()

    for epoch in range(args.epoch_begin, args.epoch_num):
        for step, motions in enumerate(data_loader): #data_loader로부터 motion을 가져온다
            model.set_input(motions) #모델의 motions_input을 가져온 모션으로 대체한다
            model.optimize_parameters() #모델의 최적와 작업을 진행한다. 이 과정을 forward를 포함한다.
            # foward를 호출한 후, G와 D의 가중치를 업데이트한다

            if args.verbose: 
                res = model.verbose()
                print('[{}/{}]\t[{}/{}]\t'.format(epoch, args.epoch_num, step, len(data_loader)), res)

        if epoch % 200 == 0 or epoch == args.epoch_num - 1:
            model.save() #epoch가 200단위로 늘어날 때마다 이를 알리고 세이브한다

        model.epoch() #스케줄러를 진행시킨다

    end_tiem = time.time()
    print('training time', end_tiem - start_time)

#전체적으로 학습을 진행하면서 리타깃된 애니메이션 파일을 만들어 저장하지는 않는다

if __name__ == '__main__':
    main()
