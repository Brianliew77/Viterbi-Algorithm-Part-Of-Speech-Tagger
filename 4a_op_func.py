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


output = open(f'{ddir}/output_probs.txt',
              'w',
              encoding="utf8")

number_of_unique_tokens = len(token_tag_counter_dic.keys())

delta = 1

for main_key, main_value in token_tag_counter_dic.items():
    for nested_key, nested_value in main_value.items():
        output.write(
            main_key + '\t' +
            nested_key + '\t' +
            str((float(nested_value) + delta) / (counter_tag_dic[nested_key] + (delta * (number_of_unique_tokens + 1)))) + '\n')
