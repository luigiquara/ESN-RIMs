import time
import torch
from data import MnistData
from networks import MnistModel, LSTM
from tqdm import tqdm
import pickle
import argparse
import numpy as np
import wandb
import os
from datetime import datetime
import math

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type = str2bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 200)

parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--hidden_size', type = int, default = 100)
parser.add_argument('--input_size', type = int, default = 1)
parser.add_argument('--model', type = str, default = 'LSTM')
parser.add_argument('--train', type = str2bool, default = True)
parser.add_argument('--num_units', type = int, default = 6)
parser.add_argument('--rnn_cell', type = str, default = 'LSTM')
parser.add_argument('--key_size_input', type = int, default = 64)
parser.add_argument('--value_size_input', type = int, default = 400)
parser.add_argument('--query_size_input', type = int, default = 64)
parser.add_argument('--num_input_heads', type = int, default = 1)
parser.add_argument('--num_comm_heads', type = int, default = 4)
parser.add_argument('--input_dropout', type = float, default = 0.1)
parser.add_argument('--comm_dropout', type = float, default = 0.1)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--key_size_comm', type = int, default = 32)
parser.add_argument('--value_size_comm', type = int, default = 100)
parser.add_argument('--query_size_comm', type = int, default = 32)
parser.add_argument('--k', type = int, default = 4)

parser.add_argument('--sched', type=str)

parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--size', type = int, default = 14)
parser.add_argument('--subset_size', type=int, default=None)
parser.add_argument('--loadsaved', type = int, default = 0)
parser.add_argument('--save_dir', type = str, default = './results/')
parser.add_argument('--load_dir', type=str, default='./results/')
parser.add_argument('--log', action='store_true')
parser.add_argument('--no_log', action='store_false', dest='log')

args = vars(parser.parse_args())

save_dir = args['save_dir'] + args['model'] + '_' + args['rnn_cell'] + '/'
if not os.path.exists(save_dir): os.mkdir(save_dir)
load_dir = args['load_dir']
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)

model = args['model']
print(f'Mode: {model}')
if args['model'] == 'LSTM':
	mode = LSTM
	print('Actually LSTM')
else:
	mode = MnistModel
	print('Another model!')




def test_model(model, loader, func):
	accuracy = 0
	loss = 0.
	iter_ctr = 0.

	model.eval()
	with torch.no_grad():
		for i in tqdm(range(loader.val_len())):
			test_x, test_y = func(i)
			test_x = model.to_device(test_x)
			test_y = model.to_device(test_y).long()
			
			probs, l = model(test_x, test_y)

			# test for nan
			assert not torch.isnan(probs).any().item(), 'Test probs are NaN!'
			assert not torch.isnan(l).any().item(), 'Test loss is NaN'

			preds = torch.argmax(probs, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()
			loss += l.item()
			iter_ctr += 1.

			# test for nan
			assert math.isnan(accuracy) == False, 'Test accuracy is NaN!'

	accuracy /= 100.0
	return loss/iter_ctr, accuracy

def train_model(model, epochs, data):
	acc=[]
	lossstats=[]
	best_acc = 0.0
	ctr = 0	

	test_acc = 0
	start_epoch = 0
	ctr = 0
	patience = 0

	if args['loadsaved']==1:
		with open(load_dir+'/accstats.pickle','rb') as f:
			acc=pickle.load(f)
		with open(load_dir+'/lossstats.pickle','rb') as f:
			losslist=pickle.load(f)
		start_epoch=len(acc)-1
		best_acc=0
		for i in acc:
			if i[0]>best_acc:
				best_acc=i[0]
		ctr=len(losslist)-1
		saved = torch.load(load_dir + '/best_model.pt')
		model.load_state_dict(saved['net'])

	if args['opt'] == 'Adam': optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
	elif args['opt'] == 'AdamW': optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])

	if args['sched'] == 'CosineAnnealingLR':
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-start_epoch)
		print(f'Using {args["sched"]} scheduler')
		
	for epoch in range(start_epoch,epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0.
		iter_ctr = 0.
		t_accuracy = 0
		norm = 0
		model.train()
		for i in tqdm(range(data.train_len())):

			# load the current batch
			start = time.time()
			iter_ctr+=1.
			inp_x, inp_y = data.train_get(i)
			inp_x = model.to_device(inp_x)
			inp_y = model.to_device(inp_y)
			time_load_data = time.time() - start

			# forward pass
			start = time.time()
			output, l = model(inp_x, inp_y)
			time_forward_pass = time.time() - start

			# test for nan
			assert not torch.isnan(output).any().item(), 'Model output is NaN!'
			assert not torch.isnan(l).any().item(), 'Loss is NaN!'

			# backward pass
			start = time.time()
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			time_backward_pass = time.time() - start

			# compute metrics
			start = time.time()
			norm += model.grad_norm()
			epoch_loss += l.item()
			preds = torch.argmax(output, dim=1)
			
			correct = preds == inp_y.long()
			t_accuracy += correct.sum().item()
			time_compute_metrics = time.time() - start

			# test for nan
			assert math.isnan(t_accuracy) == False, 'Training accuracy is NaN!'

			ctr += 1

		# adjust the learning rate
		if args['sched'] == 'CosineAnnealingLR':
			scheduler.step()
			print(f'New learning rate: {scheduler.get_last_lr()}')

		# evaluation on validation sets
		start = time.time()
		# val accuracy using 14x14 images, as in training
		v_loss14, v_accuracy14 = test_model(model, data, data.val_get)

		# val accuracy on 24x24, 19x19 and 16x16
		v_loss24, v_accuracy24 = test_model(model, data, data.val_get24)
		v_loss19, v_accuracy19 = test_model(model, data, data.val_get19)
		v_loss16, v_accuracy16 = test_model(model, data, data.val_get16)
		time_evaluation = time.time() - start

		# printing & logging
		# log metrics
		if args['log'] :wandb.log({'loss': epoch_loss/iter_ctr,
								   'val_loss14': v_loss14,
								   'val_loss16': v_loss16,
								   'val_loss19': v_loss19,
								   'val_loss24': v_loss24,
								   'val_accuracy14': v_accuracy14,
								   'val_accuracy16': v_accuracy16,
								   'val_accuracy19': v_accuracy19,
								   'val_accuracy24': v_accuracy24,
		})
		# log execution times
		if args['log']: wandb.log({'time_load_data': time_load_data,
								   'time_forward_pass': time_forward_pass,
								   'time_backward_pass': time_backward_pass,
								   'time_compute_metrics': time_compute_metrics
		})

		print('epoch_loss: {}, val_accuracy14: {}, val accuracy16: {}, val_accuracy19:{}, val_accuracy24:{}, train_acc: {}, grad_norm: {} '.format(epoch_loss/(iter_ctr), v_accuracy14, v_accuracy16, v_accuracy19, v_accuracy24, t_accuracy / 600, norm/iter_ctr))
		lossstats.append((ctr,epoch_loss/iter_ctr))
		acc.append((epoch,(v_accuracy24, v_accuracy19, v_accuracy16)))
		with open(save_dir+'/lossstats.pickle','wb') as f:
			pickle.dump(lossstats,f)
		with open(save_dir+'/accstats.pickle','wb') as f:
			pickle.dump(acc,f)

		# saving the best model
		if v_accuracy14 > best_acc:
			best_acc = v_accuracy14
			patience = 0
			
			print('new best model')
			#print('best validation accuracy ' + str(best_acc))
			print('Saving best model..')
			state = {
	    		'net': model.state_dict(),
				'epoch':epoch,
	    		'ctr':ctr,
	    		'best_acc':best_acc
	    	}
			with open(save_dir + '/best_model.pt', 'wb') as f:
				torch.save(state, f)
		# early stopping mechanism
		else:
			# patience += 1 # this means no early stopping!
			if patience > 5: 
				print('I had enough...')
				break
		
data = MnistData(args['batch_size'], (args['size'], args['size']), args['k'], args['subset_size'])
model = mode(args).cuda()
#model = mode(args)

if args['log']:
	wandb.init(
		project = 'RIM',
		config = args
	)

	save_dir += wandb.run.name

else:
	save_dir += datetime.now().strftime('%d%m%Y-%H%M%S') # if not using wandb, the name is the current time
os.mkdir(save_dir) # create the directory


if args['train']:
	train_model(model, args['epochs'], data)
	if args['log']: wandb.finish()
else:
	saved = torch.load(load_dir + '/best_model.pt')
	model.load_state_dict(saved['net'])
	v_acc14 = test_model(model, data, data.val_get)
	v_acc16 = test_model(model, data, data.val_get16)
	v_acc19 = test_model(model, data, data.val_get19)
	v_acc24 = test_model(model, data, data.val_get24)

	print(f'val_acc14: {v_acc14}\nval_acc16: {v_acc16}\nval_acc19: {v_acc19}\nval_acc24: {v_acc24}')