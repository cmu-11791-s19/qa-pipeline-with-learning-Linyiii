import sys
import json
import csv


def getlist(path):
    with open(path, 'r') as fd:
        result = fd.read().splitlines()
        for cor in range(len(result)):
            result[cor] = result[cor].strip()
    return result


# This file process the results from different models
if __name__ == '__main__':

    # read in results
    MNB_count = 'MNB_Count_pred'
    MNB_tf = 'MNB_Tfidf_pred'
    SVM_count = 'SVM_Count_pred'
    SVM_tf = 'SVM_Tfidf_pred'
    MLP_count = 'MLP_Count_pred'
    MLP_tf = 'MLP_Tfidf_pred'

    MNB_count_result = getlist(MNB_count)
    MNB_tf_result = getlist(MNB_tf)
    SVM_count_result = getlist(SVM_count)
    SVM_tf_result = getlist(SVM_tf)
    MLP_count_result = getlist(MLP_count)
    MLP_tf_result = getlist(MLP_tf)

    # sanity check
    assert (len(MNB_count_result) == len(MNB_tf_result))
    assert (len(MNB_count_result) == len(SVM_tf_result))
    assert (len(MNB_count_result) == len(MLP_tf_result))
    assert (len(MNB_count_result) == len(SVM_count_result))
    assert (len(MNB_count_result) == len(MLP_count_result))

    # get examples with all 1's, all 0's and mixtures:
    all_results = [MNB_count_result, MNB_tf_result, SVM_count_result,
                   SVM_tf_result, MLP_count_result, MLP_tf_result]
    all_zeros, all_ones, mix = {}, {}, {}  # {example index: 000000}
    for i in range(len(MNB_count_result)):
        temp = []
        # get 0/1 from each model
        for j in range(6):
            temp.append(all_results[j][i])
        reference = ''.join(temp)  # six digits that indicate model correctness
        # if reference == '000000' and len(all_zeros) < 101:
        if reference == '000000':
            all_zeros[i] = reference
        # elif reference == '111111' and len(all_ones) < 101:
        elif reference == '111111':
            all_ones[i] = reference
        # elif (not reference == '000000') and (not reference == '111111') and len(mix) < 201:
        else:
            mix[i] = reference

    # get questions
    val_questions = []
    with open("quasar-s_dev_formatted.json", 'r') as dev_fd:
        val_data = json.load(dev_fd)  # this is a dictionary
        val_questions = val_data['questions']  # this is a list

    # get examples from each category
    with open('all_zeros.csv', mode='w') as all_zeros_fd:
        all_zeros_writer = csv.writer(all_zeros_fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        all_zeros_writer.writerow(['reference', 'answer type', 'answer', 'query'])
        for j in all_zeros:
            corr = all_zeros[j]
            q = val_questions[j]
            answer_type = q['answer_type']
            answer = q['answers'][0]
            query = q['query']
            all_zeros_writer.writerow([corr, answer_type, answer, query])

    with open('all_ones.csv', mode='w') as all_ones_fd:
        all_ones_writer = csv.writer(all_ones_fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        all_ones_writer.writerow(['reference', 'answer type', 'answer', 'query'])
        for j in all_ones:
            corr = all_ones[j]
            q = val_questions[j]
            answer_type = q['answer_type']
            answer = q['answers'][0]
            query = q['query']
            all_ones_writer.writerow([corr, answer_type, answer, query])

    with open('mix.csv', mode='w') as mix_fd:
        mix_writer = csv.writer(mix_fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        mix_writer.writerow(['reference', 'answer type', 'answer', 'query'])
        for j in mix:
            corr = mix[j]
            q = val_questions[j]
            answer_type = q['answer_type']
            answer = q['answers'][0]
            query = q['query']
            mix_writer.writerow([corr, answer_type, answer, query])
