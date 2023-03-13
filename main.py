#coding=utf-8
from __future__ import print_function
import os
import argparse
from glob import glob

from PIL import Image
import tensorflow.compat.v1 as tf

from model import lowlight_enhance
from utils import *

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.65, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=3000, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=100, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./exdark_zerodce_result4', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/eval15/kind', help='directory for testing inputs')
parser.add_argument('--test_high_dir', dest='test_high_dir', default='./data/sci_after', help='directory for testing inputs')

parser.add_argument('--decom', dest='decom', default=1, help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()

def lowlight_train(lowlight_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[40:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []
    train_real_high_data = []
    train_low_data_eq = []
    train_low_data_clahe = []
    train_high_data_eq = []
    train_pre_max_channel = []




    train_low_data_eq_guide = []
    train_high_data_eq_guide = []
    train_low_data_eq_guide_weight = []
    train_low_data_eq_clahe_weight = []
    train_high_data_eq_guide_weight = []


    train_low_data_names = glob('./data/our485/low/*.png') 
    train_low_data_names.sort()
    
    train_high_data_names = glob('./data/our485/self/*.png') 
    train_high_data_names.sort()

    train_real_high_data_names = glob('./data/our485/self/*.png') 
    train_real_high_data_names.sort()
    
    # train_low_data_names[240:480]=train_real_high_data_names[240:480]
    # train_high_data_names[240:480]=train_real_high_data_names[240:480]

    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
    # for idx in range(240):

        low_im = load_images(train_low_data_names[idx])
        # low_im = gasuss_noise2(low_im)
        #low_im = white_world(low_im)
        train_low_data.append(low_im)

        high_im = load_images(train_high_data_names[idx])
        # high_im = guideFilter(high_im,high_im)
        train_high_data.append(medianBlur(high_im,winSize=1))

        real_high_im = load_images(train_real_high_data_names[idx])

        # train_high_data_max_chan = train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(high_im+0.004), 0.004)

        train_real_high_data.append(real_high_im)


        # if np.random.random()<=0.33:
        #     train_high_data_max_chan = mainFilter(np.max(high_im,axis=2,keepdims=True))
        # elif np.random.random()<=0.67:
        #     train_high_data_max_chan = meanFilter(np.max(high_im,axis=2,keepdims=True))
        # elif np.random.random()<=1:
        #     train_high_data_max_chan = (np.max(high_im,axis=2,keepdims=True))
        train_high_data_max_chan = (np.max(medianBlur(high_im,winSize=1),axis=2,keepdims=True))#(np.max(high_im,axis=2,keepdims=True))

        train_low_data_max_channel = (train_high_data_max_chan)
        # train_low_data_max_chan = np.max(high_im,axis=2,keepdims=True)
        pre_max_channel = (np.max(medianBlur(high_im,winSize=1),axis=2,keepdims=True))
 
        # weight_eq_clahe=0#sigmoid(5*(meanFilter(train_low_data_max_chan,(20,20))-0.5))
        # train_low_data_max_channel =histeq(train_low_data_max_chan)# + weight_eq_clahe * adapthisteq(train_low_data_max_chan)
        # train_low_data_max_channel = histeq(low_im[:,:,1])
        train_pre_max_channel.append(pre_max_channel[:,:,:])
        train_low_data_eq.append(train_low_data_max_channel[:,:,:])



    eval_low_data = []
    eval_high_data = []
    eval_real_high_data = []
    eval_low_data_name = glob('./data/eval15/low/*.*')
    eval_high_data_name = glob('./data/eval15/self/*.*')
    eval_real_high_data_name = glob('./data/eval15/high/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)
        
        eval_high_im = load_images(eval_high_data_name[idx])
        eval_high_data.append(eval_high_im)
        
        eval_real_high_im = load_images(eval_real_high_data_name[idx])
        eval_real_high_data.append(eval_real_high_im)

    lowlight_enhance.train(train_low_data,train_low_data_eq, eval_low_data,eval_high_data,train_high_data,train_pre_max_channel,train_real_high_data, eval_real_high_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'), eval_every_epoch=args.eval_every_epoch, train_phase="Decom")

    # lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'), eval_every_epoch=args.eval_every_epoch, train_phase="Relight")


def lowlight_test(lowlight_enhance):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_high_data_name = glob(os.path.join(args.test_high_dir) + '/*.*')


    test_low_data = []
    test_high_data = []
    print(test_low_data_name)
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_high_im = load_images(test_high_data_name[idx])

        # test_low_im = gasuss_noise(test_low_im)
        # test_low_im = medianBlur(test_low_im)

        test_low_data.append(test_low_im)
        test_high_data.append(test_high_im)


    # test_high_data=test_low_data
    lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, save_dir=args.save_dir, decom_flag=args.decom)


def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
