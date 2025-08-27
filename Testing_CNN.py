import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MyDataLoader_2D_Frequency_DOA import My2DNoiseDataset_Frequency_DOA, My2DNoiseDataset_Frequency_DOA1
from FD_MCSFANC_CNN_Model import Modified_ShufflenetV2_Frequency_DOA


BATCH_SIZE = 200


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader


def load_weigth_for_model(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location="cuda:0")
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)

    
def validate_single_epoch(model, eva_data_loader, device):
    fre_eval_acc = 0
    doa_eval_acc = 0
    both_eval_acc = 0
    eval_acc = 0
    model.eval()
    
    i = 0
    for input, fre_target, doa_target in eva_data_loader:
        input, fre_target, doa_target = input.to(device), fre_target.to(device), doa_target.to(device)
        i += 1 
        
        # Calculating the loss value
        fre_prediction, doa_prediction = model(input)

        # recording the validating loss and accuracy
        _, fre_pred = fre_prediction.max(1)
        fre_num_correct = (fre_pred == fre_target).sum().item()
        fre_acc = fre_num_correct / input.shape[0]
        fre_eval_acc += fre_acc
        
        _, doa_pred = doa_prediction.max(1)
        doa_num_correct = (doa_pred == doa_target).sum().item()
        doa_acc = doa_num_correct / input.shape[0]
        doa_eval_acc += doa_acc
        
        # 计算两个标签都匹配的准确率
        both_num_correct = ((fre_pred == fre_target) & (doa_pred == doa_target)).sum().item()
        both_acc = both_num_correct / input.shape[0]
        both_eval_acc += both_acc

    print(f"Testing Frequency Accuracy: {fre_eval_acc / i}", 
          f"Testing DOA Accuracy: {doa_eval_acc / i}", 
          f"Testing Both Accuracy: {both_eval_acc / i}")
    return both_eval_acc / i


def Test_CNN_Frequency_DOA(TESTING_DATASET_FILE, MODLE_PATH, File_sheet):

    testing_dataset = My2DNoiseDataset_Frequency_DOA(TESTING_DATASET_FILE, File_sheet)
    testing_loader = create_data_loader(testing_dataset, BATCH_SIZE)
    
    # set the model
    model = Modified_ShufflenetV2_Frequency_DOA(num_classes1=7, num_classes2=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # loading coefficients 
    load_weigth_for_model(model, MODLE_PATH)
    
    # testing model
    accuracy = validate_single_epoch(model, testing_loader, device)
    
    return accuracy


def Output_Test_Error_Samples(TESTING_DATASET_FILE, MODLE_PATH, File_sheet):
    testing_dataset = My2DNoiseDataset_Frequency_DOA1(TESTING_DATASET_FILE, File_sheet)
    
    model = Modified_ShufflenetV2_Frequency_DOA(num_classes1=7, num_classes2=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # loading coefficients
    load_weigth_for_model(model, MODLE_PATH)
    model.eval()
    
    j=0
    for i in range(len(testing_dataset)):
        audio_sample_path, input_spectorgram, fre_target, doa_target = testing_dataset[i]
        input_spectorgram = input_spectorgram.to(device)
        input_spectorgram = input_spectorgram.unsqueeze(0)
        fre_prediction, doa_prediction = model(input_spectorgram)
        fre_predict = torch.argmax(fre_prediction).item()
        doa_predict = torch.argmax(doa_prediction).item()
        
        if fre_predict == fre_target and doa_predict == doa_target:
            j += 1
        else:
            print(audio_sample_path, fre_predict, fre_target, doa_predict, doa_target) # error predicted noise
    accuracy = j/len(testing_dataset) # test_accuracy
    return accuracy