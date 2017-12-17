import re

import nltk
import numpy as np
from keras.layers import Activation, Dropout, Input, Dense
from keras.models import Model
from MemN2N import MemN2N
import glob


def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def get_data_vectors(qa_data, qa_vocab, word_index):
    stories_one_hot = []
    queries_one_hot = []
    answers_one_hot = []

    story_len = 0
    story_sentence_len = 0
    query_sentence_len = 0
    for story, query, answer in qa_data:
        story_one_hot_elem = []
        query_one_hot_elem = []
        answer_one_hot_elem = np.zeros(shape=[len(qa_vocab)])

        def get_word_one_hot(word):
            word_one_hot = np.zeros(shape=[len(qa_vocab)])
            word_one_hot[word_index[word]] = 1
            return word_one_hot

        for story_sentence in story:
            story_sentence_one_hot = []
            for word in story_sentence:
                if word in word_index:
                    story_sentence_one_hot.append(get_word_one_hot(word))
            story_one_hot_elem.append(np.array(story_sentence_one_hot))
            story_sentence_len = max(
                len(story_sentence_one_hot), story_sentence_len)
        story_len = max(len(story), story_len)

        for word in query:
            if word in word_index:
                query_one_hot_elem.append(get_word_one_hot(word))
        if answer in word_index:
            answer_one_hot_elem[word_index[answer]] = 1

        stories_one_hot.append(story_one_hot_elem)
        queries_one_hot.append(np.array(query_one_hot_elem))
        query_sentence_len = max(len(query_one_hot_elem), query_sentence_len)
        answers_one_hot.append(answer_one_hot_elem)

    return stories_one_hot, queries_one_hot, np.array(answers_one_hot), story_len, story_sentence_len, query_sentence_len


def pad_stories(stories_data, vocab, story_maxlen):
    padded_stories = []
    for stories in stories_data:
        padded_stories_elem = []
        for index in range(story_maxlen - len(stories)):
            padded_stories_elem.append(np.zeros(len(vocab)))
        for index in range(len(stories)):
            padded_stories_elem.append(stories[index])
        padded_stories.append(np.array(padded_stories_elem))

    return np.array(padded_stories)


# def main():
if __name__ == '__main__':
    data_folder = './data/en/'
    data_folder_10k = './data/en-10k/'

    qa_task = 1
    num_hops = 2

    for qa_task in range(1, 21):
        print('############################## Task: ',
              qa_task, '##############################')

        train_file = glob.glob(data_folder_10k + 'qa' +
                               str(qa_task) + '_*_train.txt')[0]
        test_file = glob.glob(data_folder + 'qa' +
                              str(qa_task) + '_*_test.txt')[0]

        train_data = get_stories(train_file)
        test_data = get_stories(test_file)
        vocab = set(nltk.wordpunct_tokenize(open(train_file).read()))
        test_vocab = set(nltk.wordpunct_tokenize(open(test_file).read()))
        vocab.update(test_vocab)
        vocab = list(vocab)
        word_index = {}
        for i, word in enumerate(vocab):
            word_index[word] = i
        train_stories_list, train_queries_list, train_answers, train_story_len, train_story_sentence_len, train_query_sentence_len = get_data_vectors(
            train_data, vocab, word_index)
        test_stories_list, test_queries_list, test_answers, test_story_len, test_story_sentence_len, test_query_sentence_len = get_data_vectors(
            test_data, vocab, word_index)

        def get_sentence_vec(sentence):
            return sentence.sum(axis=0)

        def get_story_vec(story):
            story_vec = []
            for story_sentence in story:
                story_vec.append(get_sentence_vec(story_sentence))
            return np.array(story_vec)

        def get_story_vec_list(story_list):
            story_vec_list = []
            for story in story_list:
                story_vec_list.append(get_story_vec(story))
            return story_vec_list

        def get_query_vec_list(query_list):
            query_vec_list = []
            for query in query_list:
                query_vec_list.append(get_sentence_vec(query))
            return np.array(query_vec_list)

        train_stories = get_story_vec_list(train_stories_list)
        test_stories = get_story_vec_list(test_stories_list)

        train_queries = get_query_vec_list(train_queries_list)
        test_queries = get_query_vec_list(test_queries_list)

        story_maxlen = max(train_story_len, test_story_len)
        train_stories = pad_stories(train_stories, vocab, story_maxlen)
        test_stories = pad_stories(test_stories, vocab, story_maxlen)

        for num_hops in range(2, 4):

            print('Hop: ', num_hops)

            story_m_input = Input(
                shape=(story_maxlen, len(vocab)), name='input_m')
            query_u_input = Input(shape=[len(vocab), ], name='query_u')

            output = MemN2N(output_dim=20, num_hops=num_hops)([
                story_m_input, query_u_input])

            answer = Dense(len(vocab), activation='softmax')(output)

            qa_model = Model(
                input=[story_m_input, query_u_input], output=answer)
            qa_model.compile(loss='categorical_crossentropy',
                             optimizer='rmsprop', metrics=['accuracy'])

            qa_model.fit([train_stories, train_queries], train_answers, batch_size=1, nb_epoch=120,
                         validation_data=([test_stories, test_queries], test_answers))

        print('####################################################################################')


# if __name__ == '__main__':
#     main()
