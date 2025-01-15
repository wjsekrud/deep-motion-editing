
#Modified
def get_train_A_characters():
    with open('./datasets/Mixamo/train_char_A.txt', 'r') as file:
        names = []
        for char in file:
            name = char.strip()
            names.append(name)
        return names

def get_train_B_characters():
    with open('./datasets/Mixamo/train_char_B.txt', 'r') as file:
        names = []
        for char in file:
            name = char.strip()
            names.append(name)
        return names


def get_character_names(args):
    
    characters = [get_train_A_characters(), get_train_B_characters()]
    '''
    if not args.is_train:
        """
        To run evaluation successfully, number of characters in both groups must be the same. Repeat is okay.
        """
        tmp = characters[1][args.eval_seq]
        characters[1][args.eval_seq] = characters[1][0]
        characters[1][0] = tmp
    '''
    return characters


def create_dataset(args, character_names=None):
    from datasets.combined_motion import TestData, MixedData

    if args.is_train:
        return MixedData(args, character_names)
    else:
        return TestData(args, character_names)


def get_test_set():
    with open('./datasets/Mixamo/test_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        return list


def get_train_list():
    with open('./datasets/Mixamo/train_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        return list

