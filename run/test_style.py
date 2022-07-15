import argparse
from utility.load_data import *
from utility.ml_gcn import *
from utility.engine_test import *

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', default='../data/kfashion_style')
parser.add_argument('--wordvec', default='../data/kfashion_style/custom_glove_word2vec_final.pkl')
parser.add_argument('--adj', default='../data/kfashion_style/custom_adj_final.pkl')
parser.add_argument('--n_classes', default=10)
parser.add_argument('--image-size', default=224, type=int)
parser.add_argument('-j', '--workers', default=12, type=int)
parser.add_argument('--device_ids', default=[0,1,2,3], type=int, nargs='+')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='../checkpoint/kfashion_style/model_best.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                    help='evaluate model on validation set')

def run_style_classifier():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    test_dataset = load_data(root = args.data, phase='test', inp_name=args.wordvec)

    num_classes = args.n_classes
    model = gcn_resnet101(num_classes=num_classes, t=0.03, adj_file=args.adj)
    criterion = nn.MultiLabelSoftMarginLoss()

    state = {'batch_size': args.batch_size, 'image_size': args.image_size,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = '../checkpoint/kfashion_style/'
    state['workers'] = args.workers

    if args.evaluate:
        state['evaluate'] = True
    engine = Engine(state)
    engine.learning(model, criterion, test_dataset)

if __name__ == '__main__':
    startTime = time.time()
    print("Starting..")

    run_style_classifier()
    print("[DONE] Total time spent :{:.4f}".format(time.time() - startTime))
