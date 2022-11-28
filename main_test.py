
import torch
import cv2 
import numpy as np
import os.path as osp
import time
from models.DMDNet import DMDNet
import face_alignment # pip install face-alignment or conda install -c 1adrianb face_alignment
import argparse
import os
import math
from torchvision.transforms.functional import normalize
FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda' if torch.cuda.is_available() else 'cpu')

def read_img_tensor(img_path=None, return_landmark=True): #rgb -1~1 
    Img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR or G
    if Img.ndim == 2:
        Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)  # GGG
    else:
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # RGB
    
    if Img.shape[0] < 512 or Img.shape[1] < 512:
        Img = cv2.resize(Img, (512,512), interpolation = cv2.INTER_AREA)

    ImgForLands = Img.copy()
    Img = Img.transpose((2, 0, 1))/255.0
    Img = torch.from_numpy(Img).float()
    normalize(Img, [0.5,0.5,0.5], [0.5,0.5,0.5], inplace=True)
    ImgTensor = Img.unsqueeze(0)
    SelectPred = None
    if return_landmark:
        try:
            PredsAll = FaceDetection.get_landmarks(ImgForLands)
        except:
            print('Error in detecting this face {}. Continue...'.format(img_path))
        if PredsAll is None:
            print('Warning: No face is detected in {}. Continue...'.format(img_path))
            return ImgTensor, None
        ins = 0
        if len(PredsAll)!=1:
            hights = []
            for l in PredsAll:
                hights.append(l[8,1] - l[19,1])
            ins = hights.index(max(hights))
            print('Warning: Too many faces are detected, only handle the largest one...')
        SelectPred = PredsAll[ins]
    return ImgTensor, SelectPred


def get_component_location(Landmarks, re_read=False):
    if re_read:
        ReadLandmark = []
        with open(Landmarks,'r') as f:
            for line in f:
                tmp = [float(i) for i in line.split(' ') if i != '\n']
                ReadLandmark.append(tmp)
        ReadLandmark = np.array(ReadLandmark) #
        Landmarks = np.reshape(ReadLandmark, [-1, 2]) # 68*2
    Map_LE_B = list(np.hstack((range(17,22), range(36,42))))
    Map_RE_B = list(np.hstack((range(22,27), range(42,48))))
    Map_LE = list(range(36,42))
    Map_RE = list(range(42,48))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))

    Landmarks[Landmarks>504]=504
    Landmarks[Landmarks<8]=8
    
    #left eye
    Mean_LE = np.mean(Landmarks[Map_LE],0)
    L_LE1 = Mean_LE[1] - np.min(Landmarks[Map_LE_B,1])
    L_LE1 = L_LE1 * 1.3
    L_LE2 = L_LE1 / 1.9
    L_LE_xy = L_LE1 + L_LE2
    L_LE_lt = [L_LE_xy/2, L_LE1]
    L_LE_rb = [L_LE_xy/2, L_LE2]
    Location_LE = np.hstack((Mean_LE - L_LE_lt + 1, Mean_LE + L_LE_rb)).astype(int)

    #right eye
    Mean_RE = np.mean(Landmarks[Map_RE],0)
    L_RE1 = Mean_RE[1] - np.min(Landmarks[Map_RE_B,1])
    L_RE1 = L_RE1 * 1.3
    L_RE2 = L_RE1 / 1.9
    L_RE_xy = L_RE1 + L_RE2
    L_RE_lt = [L_RE_xy/2, L_RE1]
    L_RE_rb = [L_RE_xy/2, L_RE2]
    Location_RE = np.hstack((Mean_RE - L_RE_lt + 1, Mean_RE + L_RE_rb)).astype(int)

    #nose
    Mean_NO = np.mean(Landmarks[Map_NO],0)
    L_NO1 =( np.max([Mean_NO[0] - Landmarks[31][0], Landmarks[35][0] - Mean_NO[0]])) * 1.25
    L_NO2 = (Landmarks[33][1] - Mean_NO[1]) * 1.1
    L_NO_xy = L_NO1 * 2
    L_NO_lt = [L_NO_xy/2, L_NO_xy - L_NO2]
    L_NO_rb = [L_NO_xy/2, L_NO2]
    Location_NO = np.hstack((Mean_NO - L_NO_lt + 1, Mean_NO + L_NO_rb)).astype(int)
    
    #mouth
    Mean_MO = np.mean(Landmarks[Map_MO],0)
    L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16)) * 1.1
    MO_O = Mean_MO - L_MO + 1
    MO_T = Mean_MO + L_MO
    MO_T[MO_T>510]=510
    Location_MO = np.hstack((MO_O, MO_T)).astype(int)
    return torch.cat([torch.FloatTensor(Location_LE).unsqueeze(0), torch.FloatTensor(Location_RE).unsqueeze(0), torch.FloatTensor(Location_NO).unsqueeze(0), torch.FloatTensor(Location_MO).unsqueeze(0)], dim=0)


def check_bbox(imgs, boxes):
    boxes = boxes.view(-1, 4, 4)
    imgWithBox = []
    colors = [(0, 255, 0), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    i = 0
    for img, box in zip(imgs, boxes):
        img = (img + 1)/2 * 255
        img2 = img.permute(1, 2, 0).float().cpu().flip(2).numpy().copy()
        for idx, point in enumerate(box):
            cv2.rectangle(img2, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), color=colors[idx], thickness=2)
            img3 = (torch.from_numpy(img2).cuda()/255. - 0.5) / 0.5
        cv2.imwrite('./ttt_{:02d}.png'.format(i), img2)
        i += 1



if __name__ == '__main__':
    '''
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--img_list', type=str, default='./TestExamples/TestGenericRestoration/GenericLists.txt', help='input path of lq image')
    parser.add_argument('-d', '--out_path', type=str, default='./TestExamples/Results_TestGenericRestoration', help='save path of restoration result')
    parser.add_argument('--check', action='store_true', help='save the face images with landmarks shown on them to check the performance')
    args = parser.parse_args()

    c_time = time.strftime("%m-%d_%H-%M", time.localtime()) 
    save_path = osp.join(args.out_path+'_'+c_time)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DMDNet = DMDNet().to(device)#
    weights = torch.load('./checkpoints/DMDNet.pth') 
    DMDNet.load_state_dict(weights, strict=True)

    DMDNet.eval()
    num_params = 0
    for param in DMDNet.parameters():
        num_params += param.numel()

    print('{:>8s} : {}'.format('Using device', device))
    print('{:>8s} : {:.2f}M'.format('Model params', num_params/1e6))
    torch.cuda.empty_cache()

    if not osp.exists(args.img_list):
        exit('Error in test image list')

    fp = open(args.img_list, 'r')
    lines = fp.read().split("\n")
    lines = [line.strip() for line in lines if len(line)]
    for line in lines:
        paths_split = line.split('\t')
        SP = False # default generic restoration
        if len(paths_split) == 1: # no reference, generic restoration
            lq_path = paths_split[0]
            if not osp.exists(lq_path):
                print('{} does not exist for generic restoration. Continue...'.format(lq_path))
                continue
        elif len(paths_split) == 3: # have references, specific restoration
            lq_path, id_path, land_path = paths_split
            if not osp.exists(lq_path):
                print('{} does not exist for specific restoration. Continue...'.format(lq_path))
                continue
            if len(os.listdir(id_path)) < 2:
                print('{} does not have enough high-quality references (>=2). Continue...'.format(id_path))
                continue
            if len(os.listdir(land_path)) < 2:
                print('{} does not have enough corresponding landmarks. You man run the ./TestSamples/FaceLandmarkDetection.py to obtain its landmarks...'.format(id_path))
                continue
            SP = True
        else:
            print('Error in reading the img_list. Please check the example of ./TestExamples/TestLists*.txt')
            continue
        
        
        lq, lq_landmarks = read_img_tensor(lq_path, return_landmark=True)
        if lq_landmarks is None:
            print('Error in detecting landmarks of {}. Maybe its quality is very low. Continue...'.format(lq_path))
            continue

        if args.check:
            Img = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED)
            PathSplits = osp.split(lq_path)
            CheckTmpPath = osp.join(PathSplits[0], 'TmpCheckLQLandmarks')
            os.makedirs(CheckTmpPath, exist_ok=True)
            for point in lq_landmarks[17:,0:2]:
                cv2.circle(Img, (int(point[0]), int(point[1])), 1, (0,255,0), 4)
            cv2.imwrite(osp.join(CheckTmpPath, PathSplits[1]), Img)
            print('Checking detected landmarks in {}'.format(CheckTmpPath))

        LQLocs = get_component_location(lq_landmarks)

        # check_bbox(lq, LQLocs.unsqueeze(0))

        if SP: # for specific restoration
            print('Restoring {} with specific restoration'.format(osp.basename(lq_path)))
            RefPaths = os.listdir(id_path)
            SpecificImgs = []
            SpecificLocs = []
            for path in RefPaths:
                basename = osp.basename(path)
                ref_tensor, _ = read_img_tensor(osp.join(id_path, path), return_landmark=False)
                SpecificImgs.append(ref_tensor)
                ref_landamrk_path = osp.join(land_path, basename+'.txt')
                ref_locs = get_component_location(ref_landamrk_path, re_read=True)
                SpecificLocs.append(ref_locs.unsqueeze(0))

            SpecificImgs = torch.cat(SpecificImgs, dim=0)
            SpecificLocs = torch.cat(SpecificLocs, dim=0)
            # check_bbox(SpecificImgs, SpecificLocs)
            SpMem256, SpMem128, SpMem64 = DMDNet.generate_specific_dictionary(sp_imgs = SpecificImgs.to(device), sp_locs = SpecificLocs)
            SpMem256Para = {}
            SpMem128Para = {}
            SpMem64Para = {}
            for k, v in SpMem256.items():
                SpMem256Para[k] = v
            for k, v in SpMem128.items():
                SpMem128Para[k] = v
            for k, v in SpMem64.items():
                SpMem64Para[k] = v
        else:
            print('Restoring {} with only generic restoration'.format(osp.basename(lq_path)))
            SpMem256Para, SpMem128Para, SpMem64Para = None, None, None


        with torch.no_grad():
            try:
                GenericResult, SpecificResult = DMDNet(lq = lq.to(device), loc = LQLocs.unsqueeze(0), sp_256 = SpMem256Para, sp_128 = SpMem128Para, sp_64 = SpMem64Para)
            except:
                print('There may be something wrong with the detected component locations. Continue...')
                continue
        save_generic = GenericResult * 0.5 + 0.5
        save_generic = save_generic.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        save_generic = np.clip(save_generic.float().cpu().numpy(), 0, 1) * 255.0

        check_lq = lq * 0.5 + 0.5
        check_lq = check_lq.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        check_lq = np.clip(check_lq.float().cpu().numpy(), 0, 1) * 255.0
        save_base_name = osp.basename(lq_path).split('.')[0]
        
        if SpecificResult is not None:
            save_specific = SpecificResult * 0.5 + 0.5
            save_specific = save_specific.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
            save_specific = np.clip(save_specific.float().cpu().numpy(), 0, 1) * 255.0
            cv2.imwrite(osp.join(save_path, save_base_name+'_GS.png'), np.hstack((check_lq, save_generic, save_specific)))
        else:
            cv2.imwrite(osp.join(save_path, save_base_name+'_G.png'), np.hstack((check_lq, save_generic)))
    
