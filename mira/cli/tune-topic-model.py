from mira.topic_model.trainer import TopicModelTuner
from mira.topic_model.accessibility_model import AccessibilityTopicModel
from mira.topic_model.expression_model import ExpressionTopicModel
import argparse
import anndata
import os
import logging
import joblib
logger = logging.getLogger(__name__)

def main(*, model, data, study, study_name, 
    output, valsize, cv, iters):

    tuner = TopicModelTuner(model, save_name = study_name, 
        study = study, cv = cv, iters = iters)

    tuner.train_test_split(data, train_size=1-valsize)
    tuner.tune(data)
    best_model = tuner.select_best_model(data)

    best_model.save(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Runs tuning scheme for topic model.')
    parser.add_argument('--dtype','-t',required=True, choices=['rna','atac'],
        help = 'Whether modeling RNA-seq or ATAC-seq data.')
    parser.add_argument('--model','-m',required=True,type=str,
        help='Path to topic model with learning rate pre-configured.')
    parser.add_argument('--data', '-d', required=True, type = str,
        help = 'Path to anndata of pre-processed data for training. Should contain only cells that you wish to train on, and have all columns expected by the topic model.')
    parser.add_argument('--output', '-o', required=True, type = str,
        help = 'Output path for trained model.')
    parser.add_argument('--study_name', '-s', required=True, type = str,
        help = 'Path to study. If already exists, will resume that study. If not, will start a new study and save progress here.')
    parser.add_argument('--validation_size', '-v', default=0.2, type = float,
        help = 'Proportion of cells to save for validation set.')
    parser.add_argument('-cv', type = int, default= 5, help='Cross validation folds per trail')
    parser.add_argument('--iters','-i',default=64, type = int)

    args = parser.parse_args()

    if os.path.isfile(args.study_name):
        study = TopicModelTuner.load(args.study_name)
    else:
        study = None

    if args.dtype == 'rna':
        model = ExpressionTopicModel.load(args.model)
    else:
        model = AccessibilityTopicModel.load(args.model)

    model.save(args.output)
    os.remove(args.output)

    data = anndata.read_h5ad(args.data)

    main(
        model = model,
        data = data,
        study = study,
        study_name = args.study_name,
        output = args.output,
        valsize = args.validation_size,
        cv = args.cv,
        iters = args.iters
    )