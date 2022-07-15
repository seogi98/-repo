import argparse
from utility.load_data_c import *
from utility.resnest import *
from utility.engine_test_att import *
import numpy as np

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--image-size', default=224, type=int)
parser.add_argument('-j', '--workers', default=12, type=int)
parser.add_argument('--device_ids', default=[0,1,2,3], type=int, nargs='+')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                    help='evaluate model on validation set')

def run_attribute_classifier(model_name ='category'):
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    state = {'batch_size': args.batch_size, 'image_size': args.image_size,
             'evaluate': args.evaluate}

    if model_name == 'category':
        num_classes = 21
        state['resume'] = '../checkpoint/kfashion_category/model_category_best.pth.tar'
    elif model_name == 'detail':
        num_classes = 40
        state['resume'] = '../checkpoint/kfashion_detail/model_detail_best.pth.tar'
    elif model_name == 'texture':
        num_classes = 27
        state['resume'] = '../checkpoint/kfashion_texture/model_texture_best.pth.tar'
    elif model_name == 'print':
        num_classes = 21
        state['resume'] = '../checkpoint/kfashion_print/model_print_best.pth.tar'

    test_dataset = load_data(attribute = model_name, phase='test', num_classes = num_classes)

    model = resnest50d(pretrained=False, nc=num_classes)
    state['evaluate'] = True
    criterion = nn.MultiLabelSoftMarginLoss()
    engine = Engine(state)
    a = engine.learning(model, criterion, test_dataset, model_name)
    return a
if __name__ == '__main__':
    startTime = time.time()
    print("Starting..")

    lst = []
    p1 = run_attribute_classifier(model_name='category')
    lst.append(p1)
    print("[DONE] category time spent :{:.4f}".format(time.time() - startTime))

    p2 = run_attribute_classifier(model_name='detail')
    lst.append(p2)
    print("[DONE] detail time spent :{:.4f}".format(time.time() - startTime))

    p3= run_attribute_classifier(model_name='texture')
    lst.append(p3)
    print("[DONE] texture time spent :{:.4f}".format(time.time() - startTime))

    p4 = run_attribute_classifier(model_name='print')
    lst.append(p4)
    print("[DONE] print time spent :{:.4f}".format(time.time() - startTime))

    print('Average recall:', np.mean(lst))
    print("[DONE] Total time spent :{:.4f}".format(time.time() - startTime))

