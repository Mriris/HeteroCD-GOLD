import os

from torch.utils.data import DataLoader

from models.GOLD import TripleHeteCD
from options.train_options import TrainOptions

os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
from datasets import dataset
from datasets.dataset import *
from utils.util import *
# 生成一个随机5位整数


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    val_set = dataset.Data('val',img_transform=None, root = opt.dataroot)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)
    model = TripleHeteCD(opt, is_train=False)
    model.setup(opt)
    model.load_weights("checkpoints/resunet_dual_out_mask/80_net_G.pth")
    pred_dir = "preds"
    preds = []
    labels = []
    for i, data in enumerate(val_loader):
        img_A = data[0]
        img_B = data[1]
        label = data[2]
        name = data[3]
        model.set_input(data[0],data[1],data[2],data[3],opt.gpu_ids[0]) 
        out_change = model.get_cd_pred()[-1]
        pred = torch.argmax(out_change, dim=1).cpu().numpy().astype(np.uint8)
        preds.append(pred)
        labels.append(label.cpu().numpy().astype(np.uint8))
        name = name[0]
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        save_path = os.path.join(pred_dir, name)
        print(save_path)
        cv2.imwrite(save_path, pred[0] * 255)

    preds_all = np.concatenate(preds, axis=0)
    labels_all = np.concatenate(labels, axis=0)
    hist = get_confuse_matrix(2,labels_all,preds_all)
    score = cm2score(hist)
    print(score)