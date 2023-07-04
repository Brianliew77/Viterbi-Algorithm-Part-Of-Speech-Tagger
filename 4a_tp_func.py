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

twitter_train_token_tag_file_array = twitter_train_token_tag_file.read().splitlines()


# need to handle the empty spaces, hence purge all the empty lines first
twitter_train_token_tag_file_array_new = []
for i in range(0, len(twitter_train_token_tag_file_array)):
    tag_token = twitter_train_token_tag_file_array[i].strip().split()
    if (len(tag_token) == 2):
        twitter_train_token_tag_file_array_new.append(
            twitter_train_token_tag_file_array[i])
    else:
        continue

# check the twitter_train_token_tag_file_array, loop to check the tag transitions
# for each tag transition, add to count
# counter_tag_transition_dic is in the form {(tag_t-1, tag_t-2): count}

counter_tag_transition_dic = {}

for i in range(1, len(twitter_train_token_tag_file_array_new)):
    tag_token_t_1 = twitter_train_token_tag_file_array_new[i-1].strip().split()
    tag_token_t = twitter_train_token_tag_file_array_new[i].strip().split()
    tag_t_1 = tag_token_t_1[1]
    tag_t = tag_token_t[1]
    two_tags_tuple = (tag_t_1, tag_t)
    if two_tags_tuple not in counter_tag_transition_dic:
        # if two_tags_array does not exist, add in the { two_tags_array: 1 }
        counter_tag_transition_dic[two_tags_tuple] = 1
    else:
        counter_tag_transition_dic[two_tags_tuple] = counter_tag_transition_dic[two_tags_tuple] + 1

# print(len(counter_tag_transition_dic))
# print(sum(counter_tag_transition_dic.values()))

# counter_tag_dic will count the count of each of the 25 tags given
# value will be the number of times the tags appear in the training data
# counter_tag_dic = {tag : tag count}
counter_tag_dic = {}
for official_tag in official_tags_array:
    counter_tag_dic[official_tag] = 0

for x in twitter_train_token_tag_file_array:
    tag_token = x.strip().split()
    if len(tag_token) == 2:
        tag_temp = tag_token[1]
        counter_tag_dic[tag_temp] = counter_tag_dic[tag_temp] + 1

# print(counter_tag_dic)
# to count the number of unique tokens
# token_tag_counter_dic = { token: { tag: counter }}
token_tag_counter_dic = {}
for x in twitter_train_token_tag_file_array:
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

print(len(token_tag_counter_dic))
output = open(f'{ddir}/trans_probs.txt',
              'w',
              encoding="utf8")

number_of_unique_tokens = len(token_tag_counter_dic.keys())

delta = 1

# counter_tag_transition_dic is in the form {(tag_t-1, tag_t-2): count}
# counter_tag_dic = {tag : tag count}
# need to match tag_t-1

for main_key, main_value in counter_tag_transition_dic.items():
    for nested_key, nested_value in counter_tag_dic.items():
        # if tag_t-1 matches with the tag in counter_tag_dic
        if main_key[0] == nested_key:
            # output is in the form: (tag_t-1, tag_t-2) \t count(Yt-1 = i, Yt = j)/count(Yt-1 = i)
            output.write(
                str(main_key) +
                '\t' +
                str((float(main_value) + delta) /
                    (float(nested_value) + (delta * (number_of_unique_tokens + 1))))
                + '\n'
            )
