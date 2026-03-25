import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.attentionhead import AttentionSingleBranch, AttentionMultiBranch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
try:
    import wandb
except:
    print(" W&B disabled due to init failure")
from tqdm import tqdm

HPARAM_OPT = False
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False,save_ckpt=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_ckpt = save_ckpt


    def __call__(self, epoch, val_loss, model, optimizer, ckpt_name = 'checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_best_checkpoint(val_loss, model, epoch,optimizer, ckpt_name)
            self.save_last_checkpoint(val_loss, model, epoch,optimizer, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
            self.save_last_checkpoint(val_loss, model, epoch,optimizer, ckpt_name)
        else:
            self.best_score = score
            self.save_best_checkpoint(val_loss, model, epoch,optimizer, ckpt_name)
            self.counter = 0

    def save_last_checkpoint(self, val_loss, model, epoch,optimizer, ckpt_name):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            },  ckpt_name.replace("best","last"))
        self.val_loss_min = val_loss

    def save_best_checkpoint(self, val_loss, model, epoch,optimizer, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            },  ckpt_name)
        self.val_loss_min = val_loss

    def del_last_checkpoint(self,  ckpt_name):
        try:
            os.remove(ckpt_name.replace("best","last"))
            print("removed: ",  ckpt_name.replace("best","last"))
        except:
            print("could not remove ", ckpt_name.replace("best","last"))

    def del_best_checkpoint(self, ckpt_name):
        try:
            os.remove(ckpt_name.replace("best","best"))
            print("removed: ",  ckpt_name.replace("best","best"))
        except:
            print("could not remove ",ckpt_name.replace("best","best") )

def get_pretrained_model(args,fold=0,for_ood=False,device="cpu"):
    """   
        train for a single fold
    """
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
  
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError  
    
    elif args.model_type == 'addmil':
        if args.model_size=='small':
            model_dict.update({'size_arg': 'small'})	
        model_dict.update({'additive': True})
        model = AttentionSingleBranch(**model_dict)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)

    print("loading checkpoint")
    checkpoint = torch.load(os.path.join(args.results_dir, "s_{}_best.pt".format(fold)), weights_only=False,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("checkpoint loaded")

    return model



def parse_config_file(args):
    config = wandb.config
    for key, value in wandb.config.items():
        setattr(args, key, value)
    return args


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    if args.use_wandb:  
        if args.hparamoptimisation_config:
            globals()["HPARAM_OPT"] = True
            wandb.init(config=args.hparamoptimisation_config, project=args.wandbproject, name=f"{args.exp_code}_s{cur}", dir=args.results_dir)
            args = parse_config_file(args)
        else:
            wandb.init(project=args.wandbproject, name=f"{args.exp_code}_s{cur}", dir=args.results_dir)

    print(args)
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    #Implementing Additive MIL model    
    
    elif args.model_type == 'addmil':
        if args.model_size=='small':
            model_dict.update({'size_arg': 'small'})	
        model_dict.update({'additive': True})
        model = AttentionSingleBranch(**model_dict)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)


    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    if args.use_wandb:
        wandb.run.summary['lr'] = args.lr

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        _early_stopping = EarlyStopping(patience = 5, stop_epoch=50, verbose = True, save_ckpt=(not args.del_ckpt))

    else:
        _early_stopping = None
    print('Done!')
    resume_epoch = 0
    if args.resume:
        try:
            print("loading checkpoint")
            print(os.path.join(args.results_dir, "s_{}_last.pt".format(cur)))
            checkpoint = torch.load(os.path.join(args.results_dir, "s_{}_last.pt".format(cur)), weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resume_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            model.train()
            print("not using checkpoint")
        except:
            print("No checkpoint loaded. Check the path")
    
    
    for epoch in range(resume_epoch,args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            resultdict = train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn,args.use_wandb)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                _early_stopping, writer, loss_fn, args.results_dir,args.use_wandb, optimizer)
        elif args.model_type in ['addmil']:   
            resultdict = train_loop_clam_addmil(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn,args.use_wandb)
            stop = validate_clam_addmil(cur, epoch, model, val_loader, args.n_classes, 
                _early_stopping, writer, loss_fn, args.results_dir,args.use_wandb, optimizer)
        else:
            resultdict = train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,args.use_wandb)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                _early_stopping, writer, loss_fn, args.results_dir,args.use_wandb, optimizer)

        if args.use_wandb:
            wandb.log({"epoch": epoch})

        if stop: 
            break


    checkpoint = torch.load(os.path.join(args.results_dir, "s_{}_best.pt".format(cur)), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    results_val_dict, results_test_dict, test_auc, val_auc, test_error, val_error, acc_logger = post_training(args,model,val_loader, test_loader)

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar(f'final/test_class_{i}_acc', acc, 0)

        if args.use_wandb:
            wandb.run.summary[f'final/test_class_{i}_acc'] = acc

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
        
    if args.use_wandb:
        wandb.run.summary['final/val_error'] = val_error
        wandb.run.summary['final/val_auc'] = val_auc
        wandb.run.summary['final/test_error'] = test_error
        wandb.run.summary['final/test_auc'] = test_auc
        wandb.run.summary['final/loss_overfit'] = ((val_error-resultdict["train/loss"])/ val_error) #proportion of val_error that train_error is lower
        wandb.finish()



    return results_val_dict, results_test_dict, test_auc, val_auc, 1-test_error, 1-val_error 

def post_training(args,model,val_loader, test_loader):
    """
        Post training evaluation
    """
    print("loading best checkpoint")
    print(args)
    """ Post training """
    if args.model_type in ['addmil']:
        results_val_dict, val_error, val_auc, _, _ = summary_clam_addmil(model, val_loader, args.n_classes) #Changed to results_val_dict to reflect val results
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_test_dict, test_error, test_auc, acc_logger = summary_clam_addmil(model, test_loader, args.n_classes) #Changed to results_test_dict to reflect test results
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    else: #CLAM original handling
        results_val_dict, val_error, val_auc, _, _ = summary(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_test_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    return results_val_dict, results_test_dict, test_auc, val_auc, test_error, val_error, acc_logger

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None,use_wandb=False):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    n = len(loader)
    pbar = tqdm(enumerate(loader), total = len(loader),ncols=200,desc=f"epoch {epoch}")
    for batch_idx, (data, label) in pbar:
    # for batch_idx, (data, label) in tqdm(enumerate(loader)):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
            # print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
            #     'label: {}, bag_size: {}'.format(label.item(), data.size(0)))
        # pbar.set_postfix(f"loss: {loss_value:4f}, total: {total_loss.item():4f}")
        pbar.set_postfix({"loss" : f"{loss_value:4f}", "total" : f"{total_loss.item():4f}"})

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= n
    train_error /= n
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
        if use_wandb:
            wandb.log({f'train/class_{i}_acc': acc, "epoch": epoch})

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
    
    if use_wandb:
        wandb.log({"train/loss": train_loss, "train/error": train_error, "train/clustering_loss": train_inst_loss, "epoch": epoch})
    return  {"train/loss": train_loss, "train/error": train_error, "train/clustering_loss": train_inst_loss, "epoch": epoch}

def train_loop_clam_addmil(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None,use_wandb=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.


    print('\n')
    n = len(loader)
    pbar = tqdm(enumerate(loader), total = len(loader),ncols=200,desc=f"epoch {epoch}")
    for batch_idx,(data,label) in pbar:
    # for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        #data = data.unsqueeze(0) #Add one dimension for batch size
        bag_logits, att_raw, results_dict = model(data)

        Y_hat = torch.topk(bag_logits, 1, dim = 1)[1]
        Y_prob = F.softmax(bag_logits, dim = 1) #probs = sigmoid(logits) if self.is_multilabel else softmax(logits, dim=1)

        #TODO
        acc_logger.log(Y_hat, label)
        loss = loss_fn(bag_logits, label)
        loss_value = loss.item()
     
        total_loss = loss

        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
            # print('batch {}, loss: {:.4f}, '.format(batch_idx, loss_value) + 
            #     'label: {}, bag_size: {}'.format(label.item(), data.size(0)))
        pbar.set_postfix({"loss" : f"{loss_value:4f}", "total" : f"{total_loss.item():4f}"})


        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= n
    train_error /= n
    

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

        if use_wandb:
            wandb.log({f'train/class_{i}_acc': acc, "epoch": epoch})


    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
    
    if use_wandb:
        wandb.log({"train/loss": train_loss, "train/error": train_error, "epoch": epoch})
    return {"train/loss": train_loss, "train/error": train_error, "epoch": epoch}
def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,use_wandb=False):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    n = len(loader)

    pbar = tqdm(enumerate(loader), total = len(loader),ncols=200,desc=f"epoch {epoch}")
    for batch_idx, (data, label) in pbar:
    # for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
        pbar.set_postfix({"loss" : f"{loss_value:4f}"})
          
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= n
    train_error /= n



    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

        if use_wandb:
            wandb.log({f'train/class_{i}_acc': acc, "epoch": epoch})

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    if use_wandb:
        wandb.log({"train/loss": train_loss, "train/error": train_error, "epoch": epoch})
    return {"train/loss": train_loss, "train/error": train_error, "epoch": epoch}

def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None,use_wandb=False,optimizer = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    n = len(loader)
    with torch.no_grad():
        for batch_idx, (data, label) in tqdm(enumerate(loader)):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= n
    val_loss /= n

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
    if use_wandb:
        wandb.log({"val/loss": val_loss, "val/auc": auc, "val/error": val_error, "epoch": epoch})

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, optimizer, ckpt_name = os.path.join(results_dir, "s_{}_best.pt".format(cur)))
        
        if early_stopping.early_stop:
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None,use_wandb=False,optimizer = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    n = len(loader)
    with torch.inference_mode():
        for batch_idx, (data, label) in tqdm(enumerate(loader)):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= n
    val_loss /= n

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    if use_wandb:
        wandb.log({"val/loss": val_loss, "val/auc": auc, "val/error": val_error, "val/inst_loss": val_inst_loss, "epoch": epoch})


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

        if use_wandb:
            wandb.log({f'val/class_{i}_acc': acc, "epoch": epoch})

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, optimizer, ckpt_name = os.path.join(results_dir, "s_{}_best.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam_addmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None,use_wandb=False,optimizer = None):
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    #sample_size = model.k_sample
    n = len(loader)
    with torch.no_grad():
        for batch_idx, (data, label) in tqdm(enumerate(loader)):
            data, label = data.to(device), label.to(device)
            #data = data.unsqueeze(0) #Add one dimension for batch size      
            #logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            bag_logits, att_raw, results_dict = model(data)
            Y_hat = torch.topk(bag_logits, 1, dim = 1)[1]
            Y_prob = F.softmax(bag_logits, dim = 1) #probs = sigmoid(logits) if self.is_multilabel else softmax(logits, dim=1)

            acc_logger.log(Y_hat, label)
            loss = loss_fn(bag_logits, label)
            val_loss += loss.item()

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= n
    val_loss /= n

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    if use_wandb:
        wandb.log({"val/loss": val_loss, "val/auc": auc, "val/error": val_error, "epoch": epoch})

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
            
        if use_wandb:
            wandb.log({f'val/class_{i}_acc': acc, "epoch": epoch})    

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, optimizer, ckpt_name = os.path.join(results_dir, "s_{}_best.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    n = len(loader)

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'pred_class': Y_hat.item(), 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= n

    if n_classes == 2:
        y_scores = all_probs[:,1]
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        y_scores = all_probs
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger,y_scores

def summary_clam_addmil(model, loader, n_classes):
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    n = len(loader)

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        #Add one dimension for batch size
        #data = data.unsqueeze(0)

        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            #logits, Y_prob, Y_hat, _, _ = model(data)
            logits,_, _ = model(data)
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            Y_prob = F.softmax(logits, dim = 1) #probs = sigmoid(logits) if self.is_multilabel else softmax(logits, dim=1)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'pred_class': Y_hat.item(),'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= n

    if n_classes == 2:
        y_scores =  all_probs[:, 1]
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        y_scores = all_probs
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    
    return patient_results, test_error, auc, acc_logger, y_scores
