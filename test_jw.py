import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['coord']
            cell = batch['cell']
            
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            if fast:
                pred = model(inp, coord, cell*max(scale/scale_max, 1))
            else:
                pred = batched_predict(model, inp, coord, cell*max(scale/scale_max, 1), eval_bsize) # cell clip for extrapolation
            
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
            
        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
            
    return val_res.item()

def eval_psnr(model, data_name, save_dir, scale_factor=4):
    model.eval()
    test_path = f'./datasets/{data_name}/HR'

    gt_images = sorted(glob.glob(test_path + '/*.png'))

    save_path = os.path.join(save_dir,  data_name)
    os.makedirs(save_path, exist_ok=True)
    total_psnrs = []

    for gt_path in gt_images:
        # print(gt_path)
        filename = os.path.basename(gt_path).split('.')[0] 
        gt = imageio.imread(gt_path)
        
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=2)
            gt = np.repeat(gt, 3, axis=2)
        h, w, c = gt.shape
        # new_h, new_w = h - h % self.args.size_must_mode, w - w % self.args.size_must_mode
        # gt = gt[:new_h, :new_w, :]
        gt_tensor = utils.numpy2tensor(gt).cuda()
        gt_tensor, pad = utils.pad_img(gt_tensor, 24*scale_factor)#self.args.size_must_mode*self.args.scale)
        _,_, new_h, new_w = gt_tensor.size()
        input_tensor = core.imresize(gt_tensor, scale=1/scale_factor)
        blurred_tensor = core.imresize(input_tensor, scale=scale_factor)

        with torch.no_grad():
            output = batched_predict(model, ((input_tensor - 0.5) / 0.5), scale_factor, bsize=30000)
            output = output.view(1,new_h,new_w,3).permute(0,3,1,2)
            output = output * 0.5 + 0.5

        output_img = utils.tensor2numpy(output[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        input_img = utils.tensor2numpy(blurred_tensor[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        gt_img = utils.tensor2numpy(gt_tensor[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        psnr = utils.psnr_measure(output_img, gt_img)

        canvas = np.concatenate((input_img,output_img, gt_img), 1)
        
        utils.save_img_np(canvas, '{}/{}.png'.format(save_path, filename))

        total_psnrs.append(psnr)

        
    total_psnrs = np.mean(np.array(total_psnrs))
    

    return  total_psnrs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=False)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        window_size=int(args.window),
        scale_max = int(args.scale_max),
        fast = args.fast,
        verbose=True)
    print('result: {:.4f}'.format(res))