import os
import numpy as np
from utils import convert_gold_texts2sequences, tokenize

pred_labels = np.load('../data/pred_labels.npy', allow_pickle=True)

# print(label_tokenizer.index_word)
# print(pred_labels[:3])



def convert(sentence, pred_label):
    convert_sentence = ''
    for i in range(len(pred_label)):
        if pred_label[i] == 2:
            # label == 'E'
            convert_sentence += sentence[i]
            convert_sentence += ' '
        elif pred_label[i] == 3:
            # label == 'S'
            convert_sentence += ' '
            convert_sentence += sentence[i]
            convert_sentence += ' '
        else:
            convert_sentence += sentence[i]
    return convert_sentence

def main(test_filename, test_filename_output):
    with open(test_filename, 'r', encoding='utf-8') as fr:
        test_sentences = fr.readlines()
    # print(test_sentences[:3])
    s = ''
    for i in range(len(test_sentences)):
        convert_sentence = convert(test_sentences[i].strip(), pred_labels[i])
        s += convert_sentence.strip() + '\n'
    with open(test_filename_output, 'w', encoding='utf-8') as fw:
        fw.write(s)
    


if __name__ == '__main__':
    main('../data/msr_testing/msr_test.utf8', '../data/msr_test_pred.utf8')



