"""nusanc鎵╂暎妯″瀷璁粌鑴氭湰
浣跨敤棰勮缁冪殑TransUNet鐢熸垚棰勬祴鎺╃爜锛岀劧鍚庣敤鎵╂暎妯″瀷杩涜绮剧粏鍖栧鐞?
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from model.diffusion_model import DiffusionNet, DiffusionPipeline

from train_transunet import calculate_iou, calculate_dice

# 鑾峰彇椤圭洰鏍圭洰褰?
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def calculate_sensitivity(preds, targets, smooth=1e-6):
    """璁＄畻鏁忔劅搴?鍙洖鐜?(Sensitivity/Recall)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return (tp + smooth) / (tp + fn + smooth)

def calculate_specificity(preds, targets, smooth=1e-6):
    """璁＄畻鐗瑰紓搴?(Specificity)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    tn = ((1 - preds) * (1 - targets)).sum()
    fp = (preds * (1 - targets)).sum()
    return (tn + smooth) / (tn + fp + smooth)

def calculate_accuracy(preds, targets, smooth=1e-6):
    """璁＄畻鍑嗙‘鐜?(Accuracy)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    correct = (preds == targets).sum()
    return (correct + smooth) / (targets.numel() + smooth)

def calculate_precision(preds, targets, smooth=1e-6):
    """璁＄畻绮剧‘鐜?(Precision)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def calculate_auc_roc(preds_prob, targets):
    """
    璁＄畻AUC-ROC
    preds_prob: 棰勬祴姒傜巼鍊?(0-1涔嬮棿鐨勮繛缁€?
    targets: 鐪熷疄鏍囩 (0鎴?)
    """
    try:
        from sklearn.metrics import roc_auc_score
        preds_prob_np = preds_prob.view(-1).cpu().numpy()
        targets_np = targets.view(-1).cpu().numpy()
        
        # 妫€鏌ユ槸鍚﹀彧鏈変竴涓被鍒?
        if len(np.unique(targets_np)) < 2:
            return 0.5  # 濡傛灉鍙湁涓€涓被鍒紝杩斿洖0.5
        
        auc = roc_auc_score(targets_np, preds_prob_np)
        return auc
    except Exception as e:
        print(f"璁＄畻AUC-ROC鏃跺嚭閿? {e}")
        return 0.0

def calculate_auc_pr(preds_prob, targets):
    """
    璁＄畻AUC-PR (骞冲潎绮惧害 Average Precision)
    preds_prob: 棰勬祴姒傜巼鍊?(0-1涔嬮棿鐨勮繛缁€?
    targets: 鐪熷疄鏍囩 (0鎴?)
    """
    try:
        from sklearn.metrics import average_precision_score
        preds_prob_np = preds_prob.view(-1).cpu().numpy()
        targets_np = targets.view(-1).cpu().numpy()
        
        # 妫€鏌ユ槸鍚﹀彧鏈変竴涓被鍒?
        if len(np.unique(targets_np)) < 2:
            return 0.0  # 濡傛灉鍙湁涓€涓被鍒紝杩斿洖0
        
        ap = average_precision_score(targets_np, preds_prob_np)
        return ap
    except Exception as e:
        print(f"璁＄畻AUC-PR鏃跺嚭閿? {e}")
        return 0.0

# --- 鏃╁仠鏈哄埗 (鍙傝€?train.py) ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_iou_max = -np.inf
        self.delta = delta
        self.path = path if path is not None else os.path.join(".", "weights", "diffusion_model_best.pth")
    
    def __call__(self, val_iou, model, optimizer=None, epoch=None, scheduler=None):
        score = val_iou
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_iou, model, optimizer, epoch, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_iou, model, optimizer, epoch, scheduler)
            self.counter = 0
    
    def save_checkpoint(self, val_iou, model, optimizer=None, epoch=None, scheduler=None):
        if self.verbose:
            print(f'Validation IoU increased ({self.val_iou_max:.6f} --> {val_iou:.6f}). Saving model...')
        
        # 淇濆瓨瀹屾暣鐨刢heckpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_iou': val_iou
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, self.path)
        self.val_iou_max = val_iou


class BloodVesselDataset(Dataset):
    """
    琛€绠″垎鍓叉暟鎹泦锛堟敮鎸佸姞杞介鍏堢敓鎴愮殑TransUNet棰勬祴锛?
    """
    def __init__(self, image_dir, mask_dir, transunet_pred_dir=None, transform=None, img_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transunet_pred_dir = transunet_pred_dir  # TransUNet棰勬祴缁撴灉鐩綍
        self.transform = transform
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.mask_preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = self.preprocess(image)
        mask = self.mask_preprocess(mask)
        mask = (mask > 0.5).float()
        
        # 鍔犺浇棰勫厛鐢熸垚鐨凾ransUNet棰勬祴锛堝鏋滄彁渚涗簡璺緞锛?
        transunet_pred = None
        if self.transunet_pred_dir is not None:
            # 浼樺厛灏濊瘯鍔犺浇.npy鏂囦欢
            pred_path_npy = os.path.join(self.transunet_pred_dir, img_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
            # 濡傛灉娌℃湁.npy锛屽皾璇曞姞杞藉浘鐗囨枃浠?
            pred_path_png = os.path.join(self.transunet_pred_dir, img_name)
            
            if os.path.exists(pred_path_npy):
                # 鍔犺浇.npy鏂囦欢
                transunet_pred = np.load(pred_path_npy)
                transunet_pred = torch.from_numpy(transunet_pred).float()
            elif os.path.exists(pred_path_png):
                # 鍔犺浇鍥剧墖鏂囦欢
                pred_image = Image.open(pred_path_png).convert('L')
                transunet_pred = self.mask_preprocess(pred_image)
                # 褰掍竴鍖栧埌[0,1]锛屼繚鐣欒繛缁€?
                transunet_pred = transunet_pred.float()
            else:
                print(f"璀﹀憡锛氭壘涓嶅埌棰勬祴鏂囦欢 {pred_path_npy} 鎴?{pred_path_png}")
        
        return image, mask, transunet_pred, img_name


def train_diffusion_model(resume_from_checkpoint=None, use_mixed_data=False):
    """
    璁粌鎵╂暎妯″瀷锛堜紭鍖栫増锛氫娇鐢ㄩ鍏堢敓鎴愮殑妯″瀷棰勬祴锛?
    
    Args:
        resume_from_checkpoint: 鍙€夛紝浠庢寚瀹氱殑checkpoint缁х画璁粌
        use_mixed_data: 鏄惁浣跨敤娣峰悎鏁版嵁锛坧atch+resize锛?
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"浣跨敤璁惧: {device}")
    
    # 鏁版嵁璺緞閰嶇疆锛氫娇鐢ㄧ浉瀵硅矾寰?
    if use_mixed_data:
        train_data_path = "./blood-vessel-diffusion-train"
        print(f"浣跨敤娣峰悎璁粌鏁版嵁锛坧atch+resize锛? {train_data_path}")
    else:
        train_data_path = "./blood-vessel"
        print(f"浣跨敤鍘熷鏁版嵁: {train_data_path}")
    
    train_image_dir = os.path.join(train_data_path, "image")
    train_mask_dir = os.path.join(train_data_path, "label")
    
    # 棰勬祴缁撴灉璺緞
    if use_mixed_data:
        transunet_pred_dir = os.path.join(train_data_path, "prediction")
        print(f"浣跨敤瀵瑰簲鐨勯娴嬬洰褰? {transunet_pred_dir}")
    else:
        transunet_pred_dir = os.path.join(".", "transunet_predictions_last", "train")
        print(f"浣跨敤棰勬祴鐩綍: {transunet_pred_dir}")
    
    # 妫€鏌ユ槸鍚﹀凡鐢熸垚棰勬祴
    if not os.path.exists(transunet_pred_dir):
        print(f"閿欒锛氭壘涓嶅埌棰勬祴缁撴灉鐩綍: {transunet_pred_dir}")
        if use_mixed_data:
            print("璇峰厛杩愯 scripts/preprocess/prepare_diffusion_data.py 鐢熸垚娣峰悎璁粌鏁版嵁")
        return
    
    # 瓒呭弬鏁拌缃?
    batch_size = 1
    learning_rate = 4e-5  # 闄嶄綆瀛︿範鐜囷紝绮剧粏鍖栦换鍔￠渶瑕佹洿娓╁拰鐨勪紭鍖?
    num_epochs = 500
    img_size = 512
    max_refine_step = 100  # 闄嶄綆鍒?00锛屼粠鏇村皬鐨勫櫔澹板紑濮嬶紝淇濈暀鏇村TransUNet缁撴瀯
    patience = 50  # 澧炲姞鏃╁仠鑰愬績
    start_epoch = 0  # 璧峰epoch
    
    # 鍒涘缓瀹屾暣鏁版嵁闆嗭紙鍖呭惈TransUNet棰勬祴锛?
    full_dataset = BloodVesselDataset(
        train_image_dir, 
        train_mask_dir, 
        transunet_pred_dir=transunet_pred_dir,
        img_size=(img_size, img_size)
    )
    
    # 鍒掑垎璁粌闆嗗拰楠岃瘉闆?(80% 璁粌, 20% 楠岃瘉)锛屼娇鐢ㄥ浐瀹氱瀛愮‘淇濇瘡娆￠獙璇佺浉鍚屾牱鏈?
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 鍥哄畾楠岃瘉闆嗙储寮曪紝纭繚姣忎釜epoch楠岃瘉鐩稿悓鐨勬牱鏈?
    val_sample_indices = list(range(min(20, len(val_dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"璁粌鏍锋湰鏁? {len(train_dataset)}")
    print(f"楠岃瘉鏍锋湰鏁? {len(val_dataset)}")
    print(f"浣跨敤棰勬祴鐩綍: {transunet_pred_dir}")
    if use_mixed_data:
        print("璁粌鏁版嵁鍖呭惈: patch鍒囧潡 + resize鏁版嵁")
    print("浣跨敤棰勫厛鐢熸垚鐨勬ā鍨嬮娴嬶紝鏃犻渶鍔犺浇妯″瀷")
    
    # 鍒濆鍖栨墿鏁ｆā鍨?
    print("鍒濆鍖栨墿鏁ｆā鍨?..")
    diffusion_model = DiffusionNet(
        img_size=img_size,
        in_channels=5, 
        out_channels=1
    ).to(device)
    
    # 瀹氫箟鎹熷け鍑芥暟鍜屼紭鍖栧櫒
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()  # 鐢ㄤ簬鍒嗗壊鎹熷け
    optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)
    
    # 瀹氫箟瀛︿範鐜囪皟搴﹀櫒
    # 褰撻獙璇侀泦 IoU 鍦?'patience' 涓?epoch 鍐呮病鏈夋彁鍗囨椂锛屽涔犵巼灏嗚 'factor' 缂╂斁
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    # 鍒濆鍖栨墿鏁ｇ閬?
    diffusion_pipeline = DiffusionPipeline(diffusion_model, device=device)
    
    # 鐢熸垚甯︽椂闂存埑鐨勬ā鍨嬩繚瀛樿矾寰?
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(".", "weights", f"diffusion_model_best_{timestamp}.pth")
    print(f"鏈€浣虫ā鍨嬪皢淇濆瓨涓? {best_model_path}")
    
    # 鍒濆鍖栨棭鍋滄満鍒讹紙浣跨敤鐩稿璺緞锛?
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=best_model_path)
    
    # 浠巆heckpoint鎭㈠璁粌
    if resume_from_checkpoint is not None:
        if os.path.exists(resume_from_checkpoint):
            print(f"\n浠巆heckpoint鍔犺浇鏉冮噸: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
            
            # 鍔犺浇妯″瀷鏉冮噸
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                diffusion_model.load_state_dict(checkpoint['model_state_dict'])
                
                # 璇㈤棶鏄惁缁ф壙璁粌杩涘害
                print(f"\nDetected checkpoint with training state (epoch {checkpoint.get('epoch', 0)})")
                continue_training = input("鏄惁缁ф壙璁粌杩涘害锛?y/n锛岄粯璁): ").strip().lower()
                
                if continue_training != 'n':
                    # 缁ф壙瀹屾暣璁粌鐘舵€?
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    early_stopping.best_score = checkpoint.get('best_iou', None)
                    early_stopping.val_iou_max = checkpoint.get('best_iou', -np.inf)
                    print(f"缁ф壙璁粌杩涘害锛氫粠epoch {start_epoch} 缁х画璁粌锛屾渶浣矷oU: {early_stopping.val_iou_max:.4f}")
                else:
                    # 鍙姞杞芥ā鍨嬫潈閲嶏紝浠巈poch 0寮€濮?
                    print("Loaded model weights only. Restart training from epoch 0.")
            else:
                # 鍏煎鏃ф牸寮忥紙鍙湁妯″瀷鏉冮噸锛?
                diffusion_model.load_state_dict(checkpoint)
                print("Loaded model weights successfully. Start from epoch 0.")
        else:
            print(f"Warning: checkpoint not found: {resume_from_checkpoint}. Start from scratch.")
    else:
        print("\nStart training from scratch")
    
    print("寮€濮嬭缁冩墿鏁ｆā鍨?..")
    for epoch in range(start_epoch, num_epochs):
        diffusion_model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Train')
        
        for images, true_masks, transunet_preds, _ in train_bar:
            images = images.to(device)
            true_masks = true_masks.to(device)
            transunet_preds = transunet_preds.to(device)
            
            # 鍏抽敭鏀硅繘锛氫娇鐢═ransUNet棰勬祴浣滀负"骞插噣璧风偣"锛岃€屼笉鏄疓T
            # 杩欐牱璁粌鍜屾帹鐞嗙殑鍒嗗竷灏变竴鑷翠簡锛屾秷闄omain gap
            clean_start = transunet_preds
            
            # 璁╂ā鍨嬩笓娉ㄥ涔?[0, max_refine_step] 鑼冨洿鍐呯殑鍘诲櫔锛屼笌娴嬭瘯鏃剁殑 refine_step 瀵归綈
            timesteps = torch.randint(0, max_refine_step, (images.size(0),)).to(device)
            
            # 涓篢ransUNet棰勬祴娣诲姞鍣０锛堣€屼笉鏄负GT娣诲姞鍣０锛?
            noisy_target_masks, noise = diffusion_pipeline.forward_process(
                clean_start, transunet_preds, timesteps
            )
            
            # 棰勬祴鍣０
            predicted_noise = diffusion_model(noisy_target_masks, images, transunet_preds, timesteps)
            
            # 璁＄畻鍣０ MSE 鎹熷け
            mse_loss = mse_criterion(predicted_noise, noise)
            
            # 浣跨敤棰勬祴鐨勫櫔澹拌繘琛屼竴姝ュ幓鍣紝鐒跺悗璁＄畻涓庣湡瀹炴帺鐮佺殑鍒嗗壊鎹熷け
            alpha_t = diffusion_pipeline.model.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            # 浣跨敤DDPM鍏紡鍙嶆帹骞插噣鍥惧儚
            predicted_clean = (noisy_target_masks - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            predicted_clean = torch.clamp(predicted_clean, 0, 1)
            
            # 璁＄畻鍒嗗壊鎹熷け锛圛oU鎹熷け + BCE鎹熷け锛?
            # 娉ㄦ剰锛氳繖閲屼粛鐒朵娇鐢℅T浣滀负鐩戠潱淇″彿锛岀‘淇濇ā鍨嬪涔犳纭殑鍒嗗壊
            # IoU鎹熷け
            intersection = (predicted_clean * true_masks).sum(dim=[1,2,3])
            union = predicted_clean.sum(dim=[1,2,3]) + true_masks.sum(dim=[1,2,3]) - intersection
            iou_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
            iou_loss = iou_loss.mean()
            
            # BCE鎹熷け锛堥渶瑕乴ogits锛屾墍浠ュ弽sigmoid锛?
            predicted_clean_logits = torch.log(predicted_clean + 1e-7) - torch.log(1 - predicted_clean + 1e-7)
            bce_loss = bce_criterion(predicted_clean_logits, true_masks)

            # 娣峰悎鎹熷け锛氬櫔澹版崯澶?+ 鍒嗗壊鎹熷け
            # 鍣０鎹熷け甯姪瀛︿範鎵╂暎杩囩▼锛屽垎鍓叉崯澶辩‘淇濆幓鍣悗鐨勮川閲?
            loss = mse_loss + 0.5 * iou_loss + 0.5 * bce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 楠岃瘉鐜妭 ---
        diffusion_model.eval()
        val_iou_list = []
        transunet_iou_list = []
        print("楠岃瘉涓?..")
        # 浣跨敤鍥哄畾鐨勯獙璇佹牱鏈‘淇濆彲姣旀€?
        with torch.no_grad():
            for idx in val_sample_indices:
                image, true_mask, transunet_pred, _ = val_dataset[idx]
                image = image.unsqueeze(0).to(device)
                true_mask = true_mask.unsqueeze(0).to(device)
                transunet_pred = transunet_pred.unsqueeze(0).to(device)
                
                # TransUNet棰勬祴锛堢洿鎺ヤ娇鐢ㄩ鍏堢敓鎴愮殑锛?
                # 浜屽€煎寲鍚庤绠桰oU锛堜笌train.py瀹屽叏涓€鑷达級
                initial_pred_bin = (transunet_pred > 0.5).float()
                transunet_iou = calculate_iou(initial_pred_bin, true_mask)  # train.py鐨勫嚱鏁板凡缁忚繑鍥?item()
                transunet_iou_list.append(transunet_iou)
                
                # refine_step 蹇呴』鍦ㄨ缁冭寖鍥村唴锛屼笖涓巑ax_refine_step涓€鑷?
                refined_mask = diffusion_pipeline.sample(transunet_pred, image, num_inference_steps=10, refine_step=max_refine_step)
                # 浜屽€煎寲鍚庤绠桰oU
                refined_mask_bin = (refined_mask > 0.5).float()
                iou = calculate_iou(refined_mask_bin, true_mask)  # train.py鐨勫嚱鏁板凡缁忚繑鍥?item()
                val_iou_list.append(iou)
        
        avg_val_iou = np.mean(val_iou_list)
        avg_transunet_iou = np.mean(transunet_iou_list)
        improvement = avg_val_iou - avg_transunet_iou
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, TransUNet IoU: {avg_transunet_iou:.4f}, Refined IoU: {avg_val_iou:.4f}, 鎻愬崌: {improvement:+.4f}')
        
        # 鏇存柊瀛︿範鐜囪皟搴﹀櫒
        scheduler.step(avg_val_iou)
        
        # 鏃╁仠鍒ゆ柇涓庝繚瀛樻渶浣虫ā鍨?
        early_stopping(avg_val_iou, diffusion_model, optimizer, epoch, scheduler)
        if early_stopping.early_stop:
            print("瑙﹀彂鏃╁仠鏈哄埗!")
            break
        
        # 瀹氭湡淇濆瓨锛堝寘鍚畬鏁磋缁冪姸鎬侊級
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(".", "weights", f"diffusion_model_epoch_{epoch+1}_{timestamp}.pth")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': diffusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': early_stopping.val_iou_max,
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, model_save_path)
            print(f"鍛ㄦ湡鎬т繚瀛? {model_save_path}")
    
    # 鍔犺浇鏈€浣虫ā鍨嬪弬鏁?
    if os.path.exists(early_stopping.path):
        checkpoint = torch.load(early_stopping.path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            diffusion_model.load_state_dict(checkpoint)
    
    final_model_path = os.path.join(".", "weights", f"diffusion_model_final_{timestamp}.pth")
    final_checkpoint = {
        'epoch': num_epochs - 1,
        'model_state_dict': diffusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': early_stopping.val_iou_max,
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"璁粌缁撴潫锛屾渶缁堟ā鍨嬪凡淇濆瓨: {final_model_path}")
    print(f"鏈€浣虫ā鍨? {best_model_path}")


def test_diffusion_model(custom_pred_dir=None, custom_data_dir=None):
    """
    娴嬭瘯鎵╂暎妯″瀷骞惰緭鍑鸿瘎浼版寚鏍囷紙浼樺寲鐗堬細浣跨敤棰勫厛鐢熸垚鐨凾ransUNet棰勬祴锛?
    
    Args:
        custom_pred_dir: 鍙€夛紝鑷畾涔夐娴嬬洰褰?
        custom_data_dir: 鍙€夛紝鑷畾涔夋暟鎹洰褰?
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"浣跨敤璁惧: {device}")
    
    # 鍒濆鍖栨墿鏁ｆā鍨?
    diffusion_model = DiffusionNet(
        img_size=512,
        in_channels=5,
        out_channels=1
    ).to(device)
    
    # 妫€鏌ユ槸鍚﹀瓨鍦ㄨ缁冨ソ鐨勬ā鍨嬶紙浣跨敤鐩稿璺緞锛?
    model_path = os.path.join(".", "weights", "diffusion_model_final.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(".", "weights", "diffusion_model_best.pth")
        
    if os.path.exists(model_path):
        diffusion_pipeline = DiffusionPipeline(diffusion_model, device=device)
        checkpoint = torch.load(model_path, map_location=device)
        
        # 鍏煎鏂版棫鏍煎紡
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_keys = set(diffusion_model.state_dict().keys())
            state_dict_filtered = {k: v for k, v in checkpoint.items() if k in model_keys}
            diffusion_model.load_state_dict(state_dict_filtered, strict=False)
        
        print(f"鎴愬姛鍔犺浇妯″瀷鏉冮噸: {model_path}")
    else:
        print("鏈壘鍒拌缁冨ソ鐨勬ā鍨嬶紝鏃犳硶杩涜娴嬭瘯璇勪及")
        return
    
    diffusion_model.eval()
    
    # 鍔犺浇娴嬭瘯闆嗭紙浣跨敤鐩稿璺緞锛?
    if custom_data_dir is not None:
        test_data_path = custom_data_dir
        print(f"浣跨敤鑷畾涔夋暟鎹洰褰? {test_data_path}")
    else:
        test_data_path = "./blood-vessel"
        print(f"浣跨敤榛樿鏁版嵁鐩綍: {test_data_path}")
    
    test_image_path = os.path.join(test_data_path, "test", "image")
    test_mask_path = os.path.join(test_data_path, "test", "label")
    
    # TransUNet棰勬祴缁撴灉璺緞锛堜娇鐢ㄧ浉瀵硅矾寰勶級
    if custom_pred_dir is not None:
        transunet_pred_dir = custom_pred_dir
        print(f"浣跨敤鑷畾涔夐娴嬬洰褰? {transunet_pred_dir}")
    else:
        # 濡傛灉浣跨敤浜嗚嚜瀹氫箟鏁版嵁鐩綍锛屽皾璇曚娇鐢ㄥ搴旂殑棰勬祴鐩綍
        if custom_data_dir is not None and "diffusion" in custom_data_dir:
            transunet_pred_dir = os.path.join(custom_data_dir, "test", "prediction")
            print(f"浣跨敤鏁版嵁闆嗗搴旂殑棰勬祴鐩綍: {transunet_pred_dir}")
        else:
            transunet_pred_dir = os.path.join(".", "transunet_predictions", "test")
            print(f"浣跨敤榛樿棰勬祴鐩綍: {transunet_pred_dir}")
    
    # 妫€鏌ユ槸鍚﹀凡鐢熸垚TransUNet棰勬祴
    if not os.path.exists(transunet_pred_dir):
        print(f"閿欒锛氭壘涓嶅埌TransUNet棰勬祴缁撴灉鐩綍: {transunet_pred_dir}")
        print("璇峰厛杩愯 scripts/inference/generate_transunet_predictions.py 鐢熸垚棰勬祴缁撴灉")
        return
    
    test_dataset = BloodVesselDataset(
        test_image_path, 
        test_mask_path, 
        transunet_pred_dir=transunet_pred_dir,
        img_size=(512, 512)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"娴嬭瘯闆嗘牱鏈暟: {len(test_dataset)}")
    
    metrics = {
        'iou': [], 'dice': [], 'precision': [], 'sensitivity': [], 
        'specificity': [], 'accuracy': [], 'auc_roc': [], 'auc_pr': []
    }
    
    print("寮€濮嬫ā鍨嬭瘎浼?..")
    with torch.no_grad():
        for images, true_masks, transunet_preds, name in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            true_masks = true_masks.to(device)
            transunet_preds = transunet_preds.to(device)
            
            # 鎵╂暎绮剧粏鍖栵細浣跨敤鏀硅繘鍚庣殑閲囨牱閫昏緫
            # refine_step 璁剧疆涓?100锛屼笌璁粌鏃剁殑max_refine_step瀵归綈
            refined_result = diffusion_pipeline.sample(transunet_preds, images, num_inference_steps=20, refine_step=100)
            
            # 淇濈暀杩炵画鍊肩敤浜庤绠桝UC
            refined_result_prob = refined_result.clone()
            
            # 浜屽€煎寲鍚庤绠楁寚鏍囷紙涓巘rain.py瀹屽叏涓€鑷达級
            refined_result_bin = (refined_result > 0.5).float()
            metrics['iou'].append(calculate_iou(refined_result_bin, true_masks))  # 宸茬粡鏄?item()
            metrics['dice'].append(calculate_dice(refined_result_bin, true_masks))  # 宸茬粡鏄?item()
            metrics['precision'].append(calculate_precision(refined_result_bin, true_masks).item())
            metrics['sensitivity'].append(calculate_sensitivity(refined_result_bin, true_masks).item())
            metrics['specificity'].append(calculate_specificity(refined_result_bin, true_masks).item())
            metrics['accuracy'].append(calculate_accuracy(refined_result_bin, true_masks).item())
            
            # 浣跨敤杩炵画姒傜巼鍊艰绠桝UC
            metrics['auc_roc'].append(calculate_auc_roc(refined_result_prob, true_masks))
            metrics['auc_pr'].append(calculate_auc_pr(refined_result_prob, true_masks))
    
    # 鎵撳嵃骞冲潎鎸囨爣
    print("\n" + "="*60)
    print("鎵╂暎妯″瀷璇勪及缁撴灉".center(60))
    print("="*60)
    print(f"IoU (Intersection over Union):     {np.mean(metrics['iou']):.4f}")
    print(f"Dice 绯绘暟 (Dice Coefficient):      {np.mean(metrics['dice']):.4f}")
    print(f"绮剧‘鐜?(Precision):                {np.mean(metrics['precision']):.4f}")
    print(f"鏁忔劅搴?鍙洖鐜?(Sensitivity/Recall): {np.mean(metrics['sensitivity']):.4f}")
    print(f"鐗瑰紓搴?(Specificity):              {np.mean(metrics['specificity']):.4f}")
    print(f"鍑嗙‘鐜?(Accuracy):                 {np.mean(metrics['accuracy']):.4f}")
    print(f"AUC-ROC:                           {np.mean(metrics['auc_roc']):.4f}")
    print(f"AUC-PR (Average Precision):        {np.mean(metrics['auc_pr']):.4f}")
    print("="*60)
    
    # 鐢熸垚鍙鍖栫粨鏋?
    print("\n鐢熸垚棰勬祴鍙鍖栫粨鏋?..")
    visualize_diffusion_results(diffusion_pipeline, test_loader, device, num_samples=len(test_dataset))


def visualize_diffusion_results(diffusion_pipeline, test_loader, device, num_samples=5, save_dir=None):
    """
    鍙鍖栨墿鏁ｆā鍨嬬殑棰勬祴缁撴灉锛堜紭鍖栫増锛氫娇鐢ㄩ鍏堢敓鎴愮殑TransUNet棰勬祴锛?
    鏄剧ず锛氬師濮嬪浘鍍忋€佺湡瀹炴帺鐮併€乀ransUNet棰勬祴銆佹墿鏁ｆā鍨嬬簿缁嗗寲鍚庣殑缁撴灉
    """
    if save_dir is None:
        save_dir = os.path.join(".", "output", "diffusion_visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    diffusion_pipeline.model.eval()
    
    # 閫傞厤灏忔牱鏈祴璇曢泦
    actual_num_samples = min(num_samples, len(test_loader.dataset))
    if actual_num_samples == 0:
        print("璀﹀憡锛氭祴璇曢泦涓虹┖锛屾棤娉曠敓鎴愬彲瑙嗗寲")
        return
    
    # 闅忔満閫夋嫨鏍锋湰锛堜絾涓嶈秴杩囧疄闄呮暟閲忥級
    if actual_num_samples < len(test_loader.dataset):
        indices = np.random.choice(len(test_loader.dataset), actual_num_samples, replace=False)
    else:
        indices = list(range(actual_num_samples))
    
    with torch.no_grad():
        fig, axes = plt.subplots(actual_num_samples, 4, figsize=(20, 5*actual_num_samples))
        if actual_num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            image, mask, transunet_pred, img_name = test_loader.dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            mask_tensor = mask.unsqueeze(0).to(device)
            transunet_pred_tensor = transunet_pred.unsqueeze(0).to(device)
            
            # TransUNet棰勬祴锛堢洿鎺ヤ娇鐢ㄩ鍏堢敓鎴愮殑锛?
            transunet_pred_bin = (transunet_pred_tensor > 0.5).float().squeeze().cpu().numpy()
            
            # 鎵╂暎妯″瀷绮剧粏鍖?
            refined_pred = diffusion_pipeline.sample(transunet_pred_tensor, image_tensor, num_inference_steps=20, refine_step=100)
            refined_pred_bin = (refined_pred > 0.5).float().squeeze().cpu().numpy()
            
            # 澶勭悊鍥惧儚鐢ㄤ簬鏄剧ず
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 0.5 + 0.5).clip(0, 1)  # 鍙嶅綊涓€鍖?
            mask_np = mask.squeeze().numpy()
            
            # 缁樺埗鍥涘垪锛氬師鍥俱€佺湡瀹炴帺鐮併€乀ransUNet棰勬祴銆佹墿鏁ｆā鍨嬬粨鏋?
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f"鍘熷鍥惧儚: {img_name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title("鐪熷疄鎺╃爜")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(transunet_pred_bin, cmap='gray')
            axes[i, 2].set_title("TransUNet棰勬祴")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(refined_pred_bin, cmap='gray')
            axes[i, 3].set_title("Diffusion refined")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'diffusion_prediction_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"鍙鍖栫粨鏋滃凡淇濆瓨鑷? {save_path}")
        plt.close()


if __name__ == "__main__":
    print("閫夋嫨杩愯妯″紡:")
    print("1. 璁粌鎵╂暎妯″瀷锛堜粠澶村紑濮嬶級")
    print("2. Train diffusion model (resume from checkpoint)")
    print("3. Test diffusion model (patch inference + stitching)")
    print("4. 鐢熸垚TransUNet棰勬祴")
    print("5. Prepare mixed training data (patch + resize)")
    
    choice = input("璇疯緭鍏ラ€夋嫨 (1, 2, 3, 4 鎴?5): ")
    
    if choice == "1":
        # 璇㈤棶鏄惁浣跨敤娣峰悎鏁版嵁
        print("\n閫夋嫨璁粌鏁版嵁:")
        print("1. Original data (blood-vessel/train)")
        print("2. Mixed data (patch+resize, generate with option 5 first)")
        data_choice = input("璇疯緭鍏ラ€夋嫨 (1 鎴?2锛岄粯璁や负1): ").strip()
        
        use_mixed = (data_choice == "2")
        train_diffusion_model(use_mixed_data=use_mixed)
        
    elif choice == "2":
        print("\n鍙敤鐨刢heckpoint:")
        weights_dir = "./weights"
        if os.path.exists(weights_dir):
            checkpoints = [f for f in os.listdir(weights_dir) if f.startswith('diffusion_model')]
            for i, ckpt in enumerate(checkpoints):
                print(f"  {i+1}. {ckpt}")
        
        ckpt_path = input("\n璇疯緭鍏heckpoint璺緞锛堟垨鐩存帴杈撳叆缂栧彿锛? ").strip()
        
        # 濡傛灉杈撳叆鐨勬槸缂栧彿
        if ckpt_path.isdigit():
            idx = int(ckpt_path) - 1
            if 0 <= idx < len(checkpoints):
                ckpt_path = os.path.join(weights_dir, checkpoints[idx])
            else:
                print("Invalid index")
                exit()
        # 濡傛灉杈撳叆鐨勬槸鐩稿璺緞锛岃ˉ鍏?
        elif not ckpt_path.startswith('.'):
            ckpt_path = os.path.join(weights_dir, ckpt_path)
        
        # 璇㈤棶鏄惁浣跨敤娣峰悎鏁版嵁
        print("\n閫夋嫨璁粌鏁版嵁:")
        print("1. Original data (blood-vessel/train)")
        print("2. Mixed data (patch+resize)")
        data_choice = input("璇疯緭鍏ラ€夋嫨 (1 鎴?2锛岄粯璁や负1): ").strip()
        
        use_mixed = (data_choice == "2")
        train_diffusion_model(resume_from_checkpoint=ckpt_path, use_mixed_data=use_mixed)
        
    elif choice == "3":
        print("\n浣跨敤patch鎺ㄧ悊妯″紡娴嬭瘯")
        print("灏嗗鍘熷浘鍒噋atch棰勬祴锛岀劧鍚庢嫾鎺ュ洖鏁村浘")
        print("璇疯繍琛? python scripts/inference/test_diffusion_patch.py")
        
    elif choice == "4":
        print("Run scripts/inference/generate_transunet_predictions.py to generate predictions")
        print("鍛戒护: python scripts/inference/generate_transunet_predictions.py")
        
    elif choice == "5":
        print("Run scripts/preprocess/prepare_diffusion_data.py to generate mixed training data")
        print("鍛戒护: python scripts/preprocess/prepare_diffusion_data.py")
        print("This will generate a mixed dataset with patch and resize samples")
        
    else:
        print("鏃犳晥閫夋嫨")



