import os
import logging
import tensorflow as tf
from model import BiLSTM_CRF
from data_load import load_data, get_batch
from tqdm import tqdm
from hparams import Hparams as hp
from utils import conlleval


logging.basicConfig(level=logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def evaluate(data, sess, model, epoch=None):
    # seqs_list = []
    labels_pred = []
    label_references = []
    # (sents, tags) = data
    label2tag = []
    label2tag = {label: tag for tag, label in hp.tag2label.items()}
    # for tag, label in hp.tag2label.items():
    #     label2tag[label] = tag
    for seqs, labels, seqs_len in get_batch(data, hp.batch_size, hp.vocab_path, hp.tag2label, shuffle=False):
        _logits, _transition_params  = sess.run([logits, transition_params],
                                 feed_dict={
                                     model.sent_input: seqs,
                                     model.label: labels,
                                     model.sequence_length: seqs_len
                                 })
        # seqs_list.extend(seqs)
        label_references.extend(labels)
        for logit, seq_len in zip(_logits, seqs_len):
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], _transition_params)
            labels_pred.append(viterbi_seq)
        # print(seqs_list)
        # print(label_references)
    model_pred = []
    epoch_num = str(epoch) if epoch != None else 'test'
    if not os.path.exists(hp.result_path): os.mkdir(hp.result_path)
    with open(hp.result_path+'results_epoch_'+(epoch_num), 'w', encoding='utf-8') as fw:
        for label_pred, (sent, tag) in zip(labels_pred, data):
            fw.write(''.join(sent)+'\n')
            fw.write(''.join(tag)+'\n')
            tag_pred = [label2tag[i] for i in label_pred]
            fw.write(''.join(tag_pred)+'\n')
            sent_res = []
            if len(label_pred) != len(sent):
                print(sent)
                print(len(label_pred))
                print(len(sent))
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_pred[i]])
            model_pred.append(sent_res)
    # label_path = os.path.join(hp.result_path, 'label_' + epoch_num)
    # metric_path = os.path.join(hp.result_path, 'result_metric_' + epoch_num)
    result = conlleval(model_pred)
    print(result)
    # print(len(label_pred))





if __name__ == '__main__':
    logging.info("Loading data")
    train_data = load_data(hp.prepro_dir + 'train.utf8', name="train")
    test_data = load_data(hp.prepro_dir + 'test.utf8', name='test')
    model = BiLSTM_CRF()
    loss, train_op, global_step = model.train()
    logits, transition_params = model.eval()
    logging.info("Graph loaded")
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        ckpt = tf.train.latest_checkpoint(hp.log_dir)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.global_variables_initializer())
        else:
            logging.info("Restore model from {}".format(ckpt))
            saver.restore(sess, ckpt)
        last_save_steps = 0
        for i in range(hp.epoch):
            logging.info("###### epoch {}".format(i))
            for seqs, label, seqs_len in get_batch(train_data, hp.batch_size, hp.vocab_path, hp.tag2label, shuffle=True):
                _loss, _, _gs = sess.run([loss, train_op, global_step],
                                         feed_dict={
                                             model.sent_input: seqs,
                                             model.label: label,
                                             model.sequence_length: seqs_len
                                         })
                logging.info("Global Steps: {}, loss: {}".format(_gs, _loss))
                if _gs - last_save_steps >= hp.steps_per_save:
                    last_save_steps = _gs
                    logging.info("save models")
                    model_name = "%s_E%02dL%.2f" % (model.name, i, _loss)
                    ckpt_name = os.path.join(hp.log_dir, model_name)
                    saver.save(sess, ckpt_name, global_step=_gs)
                    logging.info("After training of {} steps, {} has been saved.".format(_gs, ckpt_name))
            evaluate(test_data, sess, model, epoch=i)
    logging.info("Done!")
