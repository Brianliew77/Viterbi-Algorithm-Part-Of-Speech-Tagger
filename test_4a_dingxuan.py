# Liew Wei Brian (A0239041M), Leng Jin De Joel(A0234179Y), Seah Ding Xuan(A0240134X), Chen Haoli (A0234102B)

ddir = '.'

# contains all 25 possible official tags
official_tags_file = open(f'{ddir}/twitter_tags.txt',
                          'r',
                          encoding="utf8")

# trainset of token tag respectively
twitter_train_token_tag_file = open(f'{ddir}/twitter_train.txt',
                                    'r',
                                    encoding="utf8")

official_tags_array = official_tags_file.read().splitlines()
train_data = twitter_train_token_tag_file.read().splitlines()

# to count the number of unique tokens
# token_tag_counter_dic = { token: { tag: counter }}
token_tag_counter_dic = {}
for x in train_data:
    tag_token = x.strip().split()
    if len(tag_token) == 2:
        token_temp = tag_token[0].lower()  # changed to lower
        tag_temp = tag_token[1]
        if token_temp not in token_tag_counter_dic:
            # if token_temp does not exist, add in the { token: { tag: 1 }}
            token_tag_counter_dic[token_temp] = {tag_temp: 1}
        else:
            if tag_temp in token_tag_counter_dic[token_temp]:
                # if token exists and tag exists, add to the count
                token_tag_counter_dic[token_temp][tag_temp] = token_tag_counter_dic[token_temp][tag_temp] + 1
            else:
                # if token exists but tag does not exist
                token_tag_counter_dic[token_temp][tag_temp] = 1

number_of_unique_tokens = len(token_tag_counter_dic.keys())

# counter_tag_dic will count the count of each of the 25 tags given
# value will be the number of times the tags appear in the training data
# counter_tag_dic = {tag : tag count}
counter_tag_dic = {}
for official_tag in official_tags_array:
    counter_tag_dic[official_tag] = 0

for x in train_data:
    tag_token = x.strip().split()
    if len(tag_token) == 2:
        tag_temp = tag_token[1]
        counter_tag_dic[tag_temp] = counter_tag_dic[tag_temp] + 1

# print(train_data)
# print(counter_tag_dic)

trans_count = {}  # {i:{j:}}
trans_count["START"] = {}

for i in range(len(train_data)):
    # print(train_data[i].strip().split())
    jline = train_data[i].strip().split()
    if i == 0:  # start of train data
        trans_count["START"][jline[1]] = 1
        continue
    else:
        iline = train_data[i-1].strip().split()
        if iline != [] and jline != []:
            if iline[1] not in trans_count:
                trans_count[iline[1]] = {}
            if jline[1] not in trans_count[iline[1]]:
                trans_count[iline[1]][jline[1]] = 1
            else:
                trans_count[iline[1]][jline[1]] += 1

        elif iline == []:
            if jline[1] not in trans_count["START"]:
                trans_count["START"][jline[1]] = 1
            else:
                trans_count["START"][jline[1]] += 1
        elif jline == []:
            if iline[1] not in trans_count:
                trans_count[iline[1]] = {}
                trans_count[iline[1]]["STOP"] = 1
            else:
                if "STOP" not in trans_count[iline[1]]:
                    trans_count[iline[1]]["STOP"] = 1
                else:
                    trans_count[iline[1]]["STOP"] += 1
# print(trans_count)
trans_prob = {}
delta = 1
trans_prob["START"] = {}
for jkey in counter_tag_dic:
    if jkey not in trans_prob["START"]:
        # initialise tag j prob for START
        trans_prob["START"][jkey] = 1
    if jkey not in trans_count["START"]:
        # tag j not linked to START in train data : use smoothing for unknown
        trans_prob["START"][jkey] = delta / \
            (sum(trans_count["START"].values()) *
             (delta * (number_of_unique_tokens+1)))
    else:
        # print(sum(trans_count["START"].values()))

        trans_prob["START"][jkey] = (trans_count["START"][jkey] + delta) / (sum(
            trans_count["START"].values())*(delta*(number_of_unique_tokens + 1)))

for ikey in counter_tag_dic:
    if ikey not in trans_prob:
        trans_prob[ikey] = {}
    for jkey in counter_tag_dic:
        if jkey not in trans_prob[ikey]:
            # initialise tag j prob for tag i
            trans_prob[ikey][jkey] = 1
        if jkey not in trans_count[ikey]:
            # tag j not linked to tag i in train data : use smoothing for unknown
            trans_prob[ikey][jkey] = delta / \
                (sum(trans_count[ikey].values()) *
                 (delta * (number_of_unique_tokens+1)))
        else:
            trans_prob[ikey][jkey] = (trans_count[ikey][jkey] + delta) / (sum(
                trans_count[ikey].values())*(delta*(number_of_unique_tokens + 1)))
    # add trans prob for tag i to stop state
    trans_prob[ikey]["STOP"] = 1
    if "STOP" not in trans_count[ikey]:
        # tag j not linked to START in train data : use smoothing for unknown
        trans_prob[ikey]["STOP"] = delta / \
            (sum(trans_count[ikey].values()) *
             (delta * (number_of_unique_tokens+1)))
    else:
        trans_prob[ikey]["STOP"] = (trans_count[ikey]["STOP"] + delta) / (sum(
            trans_count[ikey].values())*(delta*(number_of_unique_tokens + 1)))
# print(trans_prob)
# output = open(f'{ddir}/trans_probs_dingxuan.txt',
#               'w',
#               encoding="utf8")
# for i in counter_tag_dic:
#     for j in counter_tag_dic:
#         output.write(
#             i + '\t' +
#             j + '\t' +
#             str(float(trans_prob[i][j]))
#         )
