import sys, argparse
sys.path.append('/home/deep/mxnet/example/rcnn/MXNet2Caffe-master/')
import find_mxnet, find_caffe
import mxnet as mx
sys.path.append('/home/deep/py-faster-rcnn/caffe-fast-rcnn/python/')
sys.path.append('/home/deep/py-faster-rcnn/lib')
sys.path.append('/home/deep/py-faster-rcnn/caffe-fast-rcnn/python/caffe')
sys.path.append('/home/deep/py-faster-rcnn/caffe-fast-rcnn')
import caffe
print('path ', caffe)
parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-model',    type=str, default='/home/deep/zl/mxnet2caffe/e2e')
parser.add_argument('--mx-epoch',    type=int, default=10)
parser.add_argument('--cf-prototxt', type=str, default='/home/deep/zl/mxnet2caffe/test.prototxt')
parser.add_argument('--cf-model',    type=str, default='/home/deep/zl/mxnet2caffe/train_vgg.caffemodel')
args = parser.parse_args()

_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
net = caffe.Net(args.cf_prototxt, caffe.TRAIN)

all_keys = arg_params.keys() + aux_params.keys()
all_keys.sort()

for i_key, key_i in enumerate(all_keys):
    try:
        # print 'key_i : ',key_i
        if 'label' in key_i:
            continue
        if 'data' in key_i:
            #print 'data'
            pass
        elif '_weight' in key_i:
            print("weights shape ",arg_params[key_i].asnumpy().shape)
            if '_weight_test' in key_i:
                key_caffe = key_i.replace('_weight_test', '')
                net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
            else:
                key_caffe = key_i.replace('_weight', '')
                net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
        elif '_bias' in key_i:
            print("bias shape ", arg_params[key_i].asnumpy().shape)
            if '_bias_test' in key_i:
                key_caffe = key_i.replace('_bias_test', '')
                net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
            else:
                key_caffe = key_i.replace('_bias', '')
                net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
        elif 'bn_gamma' in key_i:
            key_caffe = key_i.replace('bn_gamma', 'scale')
            net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
        elif '_gamma' in key_i:  # for prelu
            key_caffe = key_i.replace('_gamma', '')
            assert (len(net.params[key_caffe]) == 1)
            net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
        elif '_beta' in key_i:
            key_caffe = key_i.replace('bn_beta', 'scale')
            net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
        elif '_moving_mean' in key_i:
            key_caffe = key_i.replace('_moving_mean', '')
            net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
            net.params[key_caffe][2].data[...] = 1
        elif '_moving_var' in key_i:
            key_caffe = key_i.replace('_moving_var', '')
            net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
            net.params[key_caffe][2].data[...] = 1
        else:
            sys.exit("Warning!  Unknown mxnet:{}".format(key_i))

        print("% 3d | %s -> %s, initialized."
              % (i_key, key_i.ljust(40), key_caffe.ljust(30)))

    except KeyError:
        print("\nWarning!  key error mxnet:{}".format(key_i))

        # ------------------------------------------
# Finish
net.save(args.cf_model)
print("\n- Finished.\n")


