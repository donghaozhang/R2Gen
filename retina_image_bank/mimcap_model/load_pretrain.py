import torch
import torch.nn as nn
import torch.optim as optim

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

load_model_flag = True
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
    print(args)
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)
# output = self.model(images, mode='sample')
# # print('output', output)
# #print(self.model.)
# reports = self.test_dataloader.tokenizer.decode_batch(output.cpu().numpy())
# ground_truths = self.test_dataloader.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())