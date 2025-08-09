import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils

c_ml1m = 0.015283691692054992
c_pin = 0.04512896826434805

def hyperbolic_dist(u, v, c=1.0):
    """
    Compute hyperbolic distance in the Poincaré ball model with curvature c.
    u, v: tensors of shape (..., emb_dim)
    c: positive float (curvature)
    Returns: distance tensor of shape (...)
    """
    # Compute norms of embeddings
    u_norm_sq = torch.sum(u ** 2, dim=-1)
    v_norm_sq = torch.sum(v ** 2, dim=-1)
    
    # Compute inner product and then squared Euclidean distance
    inner_prod = torch.sum(u * v, dim=-1)
    euclidean_dist_sq = u_norm_sq + v_norm_sq - 2 * inner_prod

    # Incorporate curvature into denominators and numerator
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq) + 1e-5
    x = 1 + 2 * c * euclidean_dist_sq / denom
    x = torch.clamp(x, min=1 + 1e-5)
    
    # Scale the result appropriately
    dist = (2 / torch.sqrt(torch.tensor(c, device=u.device))) * torch.acosh(x)
    return dist

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--lambda_man", 
	type=float,
	default=0.0,  
	help="manifold reg lambda")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path,weights_only=False)
	MLP_model = torch.load(config.MLP_model_path,weights_only=False)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)


if config.dataset == 'ml-1m':
	dataset_curvature = c_ml1m
elif config.dataset == 'pinterest-20':
	dataset_curvature = c_pin

# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, item, label in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
        


		model.zero_grad()
        
		prediction = model(user, item)
        
		if args.lambda_man>0:
            
		 user_emb = model.embed_user_GMF(user)

		 dist_u = hyperbolic_dist(user_emb.unsqueeze(1), user_emb.unsqueeze(1), c=dataset_curvature)

		 item_emb = model.embed_item_GMF(item)

		 dist_i = hyperbolic_dist(item_emb.unsqueeze(1), item_emb.unsqueeze(1), c=dataset_curvature)


		 distsq = dist_u**2 + dist_i**2#dist=torch.sqrt(dist_u**2 + dist_i**2)
         
		 #distsq = dist_i**2

		 aff = torch.exp(-distsq)

		 prediction = model(user, item)

		 logits_dist = torch.cdist(prediction.reshape(-1,1),prediction.reshape(-1,1))

		 lap = aff*logits_dist

		 man_reg = lap.sum()
            
		else:
		 man_reg = 0
        
		loss = loss_function(prediction, label)+args.lambda_man*man_reg
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		count += 1

	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model, 
				'{}{}.pth'.format(config.model_path, config.model))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))
