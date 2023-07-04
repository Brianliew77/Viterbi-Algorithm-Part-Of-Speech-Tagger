# Liew Wei Brian (A0239041M), Leng Jin De Joel(A0234179Y), Seah Ding Xuan(A0240134X), Chen Haoli (A0234102B)

# Implement the six functions below

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):

    # predictions_2a is in the format ['tears\tN\t0.00046210720887245846', 'incredible\tA\t0.0011957757150975563']
    predictions_2a = open(in_output_probs_filename,
                          encoding="utf8").read().splitlines()

    # test_data is in the format ['withhh', 'mommma', '!', '']
    test_data = open(in_test_filename,
                     encoding="utf8").read().splitlines()

    # counter_tag_dic will count the count of each of the 25 tags given
    # value will be the number of times the tags appear in the training data
    # counter_tag_dic = {tag : tag count}
    counter_tag_dic = {}
    for prediction in predictions_2a:
        temp_token, temp_tag, prob = prediction.split('\t')
        # temp_token = temp_token.lower()
        if temp_tag not in counter_tag_dic:
            counter_tag_dic[temp_tag] = 1
        else:
            counter_tag_dic[temp_tag] = counter_tag_dic[temp_tag] + 1

    # max_prob_token_dictionary is in the form token : [tag : probability]
    # it keeps track of the maximum probability and corresponding tag for each token in the output probabilities from 2a

    max_prob_token_dictionary = {}
    for prediction in predictions_2a:
        temp_token, temp_tag, prob = prediction.split('\t')
        temp_token = temp_token.lower()
        if temp_token not in max_prob_token_dictionary:
            max_prob_token_dictionary[temp_token] = [temp_tag, prob]
        else:
            if prob > max_prob_token_dictionary[temp_token][1]:
                max_prob_token_dictionary[temp_token] = [temp_tag, prob]

    # calculate the probability of an unseen tag and select the tag with the highest probability
    # to prevent zero probabilities for unseen tags, we add a smmoothing factor delta to it
    # maximum_prob_unseen is the maximum probability of highest maximum proability of a tag in the seen tags
    # maximum_tag_unseen corresponds to the tag with the highest maximum_prob_unseen

    delta = 1
    maximum_prob_unseen = 0
    maximum_tag_unseen = ""
    for temp_tag, temp_count in counter_tag_dic.items():
        temp_prob = (0 + delta) / ((delta *
                                    (len(max_prob_token_dictionary) + 1)) + counter_tag_dic[temp_tag])
        if maximum_prob_unseen < temp_prob:
            maximum_tag_unseen = temp_tag
            maximum_prob_unseen = temp_prob

    # final_lst will correspond to the test data order
    # loop through the test data and predict a tag for each token
    # max_prob_token_dictionary selects the tag for the highest probability for that token
    # otherwise select maximum_tag_unseen

    final_lst = []
    for test_token in test_data:
        # to handle cases where inputs are empty strings
        if len(test_token) == 0:
            final_lst.append('')
        else:
            test_token = test_token.lower()
            if test_token not in max_prob_token_dictionary:
                final_lst.append(maximum_tag_unseen)
            else:
                final_lst.append(max_prob_token_dictionary[test_token][0])

    final_file = open(out_prediction_filename,
                      'w',
                      encoding="utf8")

    for temp_tag in final_lst:
        final_file.write(temp_tag + '\n')

    '''
    Q3(a)
    
    argmax_j P(y=j|x=w) = argmax_j (P(x=w|y=j)P(y=j)) / P(x=w)
    information needed:
    1) P(x=w|y=j): output probabilities from 2a
    2) P(y=j): number of times tag j appears in the training data
    3) P(x=w): number of times token w appears in the training data
    '''


def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):

    # naive_output_probs is in the format ['RT\t~\t0.3821448883833357', 'RT\tN\t0.00046210720887245846']
    naive_output_probs = open(in_output_probs_filename,
                              encoding="utf8").read().splitlines()

    # predictions_2a is in the format ['tears\tN', 'incredible\tA']
    twitter_train = open(in_train_filename,
                         encoding="utf8").read().splitlines()

    # twitter_dev_no_tag is in the format ['withhh', 'mommma', '!', '']
    twitter_dev_no_tag = open(in_test_filename,
                              encoding="utf8").read().splitlines()

    # find P(y=j), number of times tag j appears in the training data
    # counter_tag_dic will count the count of each of the 25 tags given
    # value will be the number of times the tags appear in the training data
    # counter_tag_dic = {tag : tag count}
    counter_tag_dic = {}
    for data in twitter_train:
        if len(data.split('\t')) == 2:
            temp_token, temp_tag = data.split('\t')
            # temp_token = temp_token.lower()
            if temp_tag not in counter_tag_dic:
                counter_tag_dic[temp_tag] = 1
            else:
                counter_tag_dic[temp_tag] = counter_tag_dic[temp_tag] + 1

    # find P(x=w), number of times token w appears in the training data
    # counter_token_dic will count the count of each of the tokens given
    # value will be the number of times the tokens appear in the training data
    # counter_token_dic = {token : token count}
    counter_token_dic = {}
    # su is the number of data in the trainset that is of length 2
    su = 0
    for data in twitter_train:
        if len(data.split('\t')) == 2:
            temp_token, temp_tag = data.split('\t')
            temp_token = temp_token.lower()
            if temp_token not in counter_token_dic:
                counter_token_dic[temp_token] = 1
            else:
                counter_token_dic[temp_token] = counter_token_dic[temp_token] + 1
            su += 1
    # max_prob_token_dictionary is in the form token : [tag : probability]
    # it keeps track of the maximum probability and corresponding tag for each token in the output probabilities from 2a

    max_prob_token_dictionary = {}
    for prediction in naive_output_probs:
        temp_token, temp_tag, prob = prediction.split('\t')
        # temp_token = temp_token.lower()
        # argmax_j P(y=j|x=w) = argmax_j (P(x=w|y=j)P(y=j)) / P(x=w)
        improved_prob = float(prob) * counter_tag_dic[temp_tag]/su

        temp_token = temp_token.lower()
        if temp_token not in max_prob_token_dictionary:
            max_prob_token_dictionary[temp_token] = [temp_tag, improved_prob]
        else:
            if improved_prob > max_prob_token_dictionary[temp_token][1]:
                max_prob_token_dictionary[temp_token] = [
                    temp_tag, improved_prob]

    # calculate the probability of an unseen tag and select the tag with the highest probability
    # to prevent zero probabilities for unseen tags, we add a smmoothing factor

    maximum_prob_unseen = 0
    maximum_tag_unseen = ""
    smoothing = 1
    for temp_tag, temp_count in counter_tag_dic.items():
        # P(y=j|x=w) = ( P(x=w|y=j) P(y=j) + smoothing ) / ( P(x=w) + smoothing * (number of words + 1) )
        #      since     P(x=w|y=j) = 0                      P(x=w) = 0
        # we can simplify the equation to
        # P(y=j|x=w) = ( smoothing ) / ( smoothing * (number of words + 1) )
        # hence P(y=j|x=w)P(y=j) = ( smoothing ) / ( smoothing * (number of words + 1) ) * (no. of tag j)/(total no. of tags)
        temp_prob = (smoothing / (smoothing *
                                  (len(max_prob_token_dictionary) + 1) + counter_tag_dic[temp_tag])) * counter_tag_dic[temp_tag]/su

        if maximum_prob_unseen < temp_prob:
            maximum_tag_unseen = temp_tag
            maximum_prob_unseen = temp_prob
# (0.01 / (0.01 * (len(pwj) + 1) + tag_counter[tag] )) * tag_counter[tag] / total_tag
    # loop through the test data and predict a tag for each token
    # max_prob_token_dictionary selects the tag for the highest probability for that token
    # otherwise select maximum_tag_unseen

    final_lst = []
    for test_token in twitter_dev_no_tag:
        # to handle cases where inputs are empty strings
        if len(test_token) == 0:
            final_lst.append('')
        else:
            test_token = test_token.lower()
            # test_token = lemmatizer.lemmatize(test_token)
            if test_token not in max_prob_token_dictionary:
                final_lst.append(maximum_tag_unseen)
            else:
                final_lst.append(max_prob_token_dictionary[test_token][0])

    final_file = open(out_prediction_filename,
                      'w',
                      encoding="utf8")

    for temp_tag in final_lst:
        final_file.write(temp_tag + '\n')


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):

    # in_tags_data is in the format ['@', ',', 'L', '~']
    # in_tags_data is the list of all tags
    in_tags_data = open(in_tags_filename,
                        encoding="utf8").read().splitlines()

    # in_trans_probs_data is in the format ["('~', '@')\t0.04424144725058213", "('@', '~')\t0.03315105946684894"]
    # where "('~', '@')\t0.04424144725058213" means probability of transition from tag '~' to tag '@' is 0.04424144725058213
    in_trans_probs_data = open(in_trans_probs_filename,
                               encoding="utf8").read().splitlines()

    # in_output_probs_data is in the format ['tears\tN\t0.0002775464890369137', 'incredible\tA\t0.00034100596760443307']]
    # where 'tears\tN\t0.0002775464890369137' means the probability of tag 'N' a emitting 'tears' is 0.0002775464890369137
    in_output_probs_data = open(in_output_probs_filename,
                                encoding="utf8").read().splitlines()

    # twitter_dev_no_tag is in the format ['withhh', 'mommma', '!', '']
    # where it shows the correct word associated with the tag
    twitter_dev_no_tag = open(in_test_filename,
                              encoding="utf8").read().splitlines()

    # trans_prob_dic = {(tag1, tag2): probability of transition}
    trans_prob_dic = {}
    for line in in_trans_probs_data:
        tup, prob = line.split("\t")
        trans_prob_dic[tup] = prob

    # Add start of sentence probabilities to the transition probabilities dictionary
    # 1 / len(in_tags_data) assumes a uniform distribution over all possible tags
    for tag in in_tags_data:
        trans_prob_dic[("<s>", tag)] = 1 / len(in_tags_data)

    # output_prob_dic = {(word, tag): probability of tag a emitting word}
    output_prob_dic = {}
    for line in in_output_probs_data:
        word, tag, prob = line.split("\t")
        output_prob_dic[(word, tag)] = prob

    # test_words contain all the test words
    test_words = []
    for line in twitter_dev_no_tag:
        if line.strip():
            test_words.append(line.strip())

    # Create an empty trellis for each token in the test data
    # vertibi_trellis[i] = {'Y': {'prob': 0, 'prev': None}, '#': {'prob': 0, 'prev': None}, where i is the column number
    vertibi_trellis = []
    for i in range(0, len(test_words)):
        vertibi_trellis.append({})
        for tag in in_tags_data:
            # Initialize the probability and previous tag for each tag in the vertibi_trellis
            vertibi_trellis[i][tag] = {"prob": 0, "prev": None}

    # initialise the first column of the vertibi_trellis vertibi_trellis[0]
    # probability of transition from tag '<s>' (source) to respective tag (a source,tag1)
    for tag in in_tags_data:
        # first column in the trellis
        vertibi_trellis[0][tag]["prob"] = trans_prob_dic[(
            "<s>", tag)] * output_prob_dic.get((test_words[0], tag), 1.0)  # 0.04
    print(vertibi_trellis[0])

    # fill up the vertibi_trellis
    for t in range(1, len(test_words)):
        for tag2 in in_tags_data:
            max_prob = 0
            max_prev_tag = None
            for tag1 in in_tags_data:
                prob = float(vertibi_trellis[t-1][tag1]["prob"]) * float(trans_prob_dic.get(
                    (tag1, tag2), 1.0)) * float(output_prob_dic.get((test_words[t], tag2), 1.0))
                if float(prob) > max_prob:
                    max_prob = prob
                    max_prev_tag = tag1
            vertibi_trellis[t][tag2]["prob"] = max_prob
            vertibi_trellis[t][tag2]["prev"] = max_prev_tag

    # find the best path
    best_path = []
    max_prob = 0
    for tag in in_tags_data:
        if vertibi_trellis[-1][tag]["prob"] > max_prob:
            max_prob = vertibi_trellis[-1][tag]["prob"]
            best_path = [tag]
            prev_tag = tag
    for t in range(len(test_words)-2, -1, -1):
        best_path.insert(0, vertibi_trellis[t+1][prev_tag]["prev"])
        prev_tag = vertibi_trellis[t+1][prev_tag]["prev"]

    with open(out_predictions_filename, "w", encoding="utf8") as f:
        for tag in best_path:
            f.write(tag + "\n")


def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip()
                          for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip()
                             for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth:
            correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)


def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '.'  # your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename = f'{ddir}/twitter_dev_ans.txt'

    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename,
                  in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename,
                   in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename = f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(
        viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    # trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    # output_probs_filename2 = f'{ddir}/output_probs2.txt'

    # viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
    #                  viterbi_predictions_filename2)
    # correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    # print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')


if __name__ == '__main__':
    run()
