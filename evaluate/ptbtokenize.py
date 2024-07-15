import os
import sys
import subprocess
import tempfile
import itertools

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'
STANFORD_CORENLP = 'stanford-corenlp-4.4.0.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    @classmethod
    def tokenize(cls, corpus):
        cmd = ['java', '-cp', STANFORD_CORENLP, \
               'edu.stanford.nlp.process.PTBTokenizer', \
               '-preserveLines', '-lowerCase']

        if isinstance(corpus, list) or isinstance(corpus, tuple):
            if isinstance(corpus[0], list) or isinstance(corpus[0], tuple):
                corpus = {i: c for i, c in enumerate(corpus)}
            else:
                corpus = {i: [c, ] for i, c in enumerate(corpus)}

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        final_tokenized_corpus = {}
        image_id = [k for k, v in list(corpus.items()) for _ in range(len(v))]
        # print(image_id)
        # sentences = '\n'.join([c.replace('\n', ' ') for k, v in list(corpus.items()) for c in v]).encode()
        sentences = '\n'.join([v[0] for k, v in list(corpus.items())]).encode()
        # print(corpus, sentences)
        # print(sentences)

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname = os.path.join(os.getcwd(), 'stanford-corenlp-4.4.0')
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences)
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        # cmd.append(os.path.basename(tmp_file.name))
        cmd.append(tmp_file.name)
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode()
        lines = token_lines.split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for k, line in zip(image_id, lines):
            if not k in final_tokenized_corpus:
                final_tokenized_corpus[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                                          if w not in PUNCTUATIONS])
            final_tokenized_corpus[k].append(tokenized_caption)

        return final_tokenized_corpus
