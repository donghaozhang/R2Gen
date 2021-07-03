import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from medical_report_model import R2GenModelAugv3AbrmDanliDatav2
from torchvision import transforms
import os 

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr', 'danli_datav2'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--model', type=str, default='r2gen', help='the type of model. The default model is r2gen.')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args

save_model_flag = False
if save_model_flag:
    # Define model

    # Initialize model
    model = TheModelClass()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    save_path = '/media/hdd/donghao/imcaption/R2Gen/results/pytorch_train_load/example.pth'
    epoch = 1
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)

load_model_flag = False
if load_model_flag:
    class TheModelClass(nn.Module):
        def __init__(self):
            super(TheModelClass, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    save_path = '/media/hdd/donghao/imcaption/R2Gen/results/pytorch_train_load/example.pth'
    load_model = TheModelClass()
    print('weight of fc1 layer 1 before loading the model')
    print(load_model.state_dict()['fc1.weight'])
    checkpoint = torch.load(save_path)
    load_model.load_state_dict(checkpoint['model_state_dict'])
    print('weight of fc1 layer 1 before loading the model')
    print(load_model.state_dict()['fc1.weight'])

create_model_imbank_flag = True

if create_model_imbank_flag:
    # parse arguments
    args = parse_agrs()
    # Modify arguments 
    args.image_dir = 'data/clean_danli_datav2'
    args.model = 'r2genaugv3abrm'
    args.ann_path = 'data/danli_datav2/annotationv2_debug.json'
    args.dataset_name = 'danli_datav2'
    args.max_seq_length = 60
    args.threshold = 3
    args.batch_size = 2
    args.epochs = 1
    args.save_dir = 'results/danli_datav2'
    args.gamma = 0.1
    args.seed = 9223
    args.n_gpu = 1 
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # dirpath = '/media/hdd/donghao/imcaption/R2Gen/retina_image_bank/mimcap_model/data/danli_datav2'
    dirpath = '/media/hdd/donghao/imcaption/R2Gen/retina_image_bank/mimcap_model/data/australia_dataset/Images2'
    # dirpath = './mimcap_model/data/danli_datav2'
    # filename = '7670-Cystoid Macular Edema0.png'
    # filename1 = '7671-Subhyaloid Hemorrhage0.png'
    filename = 'AA_LE.jpg'
    # filename = '7673-Macular Pucker0.png'
    # file_list = []
    model_saved_path = '/media/hdd/donghao/imcaption/R2Gen/retina_image_bank/mimcap_model/saved_model/model_best.pth'
    # create tokenizer
    tokenizer = Tokenizer(args)
    # create model
    model = R2GenModelAugv3AbrmDanliDatav2(args=args, tokenizer=tokenizer)
    checkpoint = torch.load(model_saved_path)
    model.load_state_dict(checkpoint['state_dict'])
    final_impath = os.path.join(dirpath, filename)
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    # image size after trasnformation 
    for file in os.listdir(dirpath):
        if file.endswith(".jpg"):
            final_impath = os.path.join(dirpath, file)
            image = Image.open(final_impath)
            imbank_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
            image = imbank_transform(image)
            image = image.unsqueeze(0)
            image_batch = image.repeat(args.batch_size, 1, 1, 1)
            image_batch = image_batch.to(device)
            output = model(image_batch, mode='sample')
            reports = tokenizer.decode_batch(output.cpu().numpy())
            print('image name:', file)
            print('report prediction:', reports[0])
