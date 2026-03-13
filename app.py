from io import open
from conllu import parse_incr
import re
import warnings
from sklearn_crfsuite import CRF
import pickle
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import random
import qrcode

from googletrans import Translator

from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__, static_folder='static')

VIDEO_FOLDER = "video_files"


# ---------------------------------------------------
# SERVE VIDEO FILES
# ---------------------------------------------------

@app.route('/videos/<filename>')
def videos(filename):
    return send_from_directory(VIDEO_FOLDER, filename)


# ---------------------------------------------------
# HOME / LAUNCHER
# ---------------------------------------------------

@app.route('/')
def launcher():
    return render_template('launcher.html')


# ---------------------------------------------------
# TRANSLATOR PAGE
# ---------------------------------------------------

@app.route('/translator')
def index():
    return render_template('index.html')


# ---------------------------------------------------
# RESULT PAGE + QR GENERATION
# ---------------------------------------------------

@app.route('/result', methods=['POST'])
def result():

    result2 = request.form['Name']

    isl_text, video_result = processing(result2)

    items = {
        'Speech To Text Conversion': result2,
        'Text to Indian Sign Language': isl_text[0]
    }

    # video url used in QR
    video_url = request.host_url + "videos/" + video_result

    # generate QR
    qr_img = qrcode.make(video_url)

    qr_filename = "qr_" + video_result.replace(".mp4", ".png")

    qr_path = os.path.join("static", qr_filename)

    qr_img.save(qr_path)

    return render_template(
        "result.html",
        result=items,
        result1=video_result,
        qr=qr_filename
    )


# ---------------------------------------------------
# LEARNING DASHBOARD
# ---------------------------------------------------

@app.route('/learn')
def learn():

    words = load_words()

    return render_template(
        "learn.html",
        total=len(words)
    )


# ---------------------------------------------------
# LEVEL 1 ALPHABETS
# ---------------------------------------------------

@app.route('/learn/alphabet')
def alphabet():

    letters = []

    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

        filename = ch + ".mp4"

        if os.path.isfile(VIDEO_FOLDER + "/" + filename):

            letters.append({
                "word": ch,
                "video": "/videos/" + filename
            })

    return render_template("words.html", words=letters)


# ---------------------------------------------------
# LEVEL 2 BASIC WORDS
# ---------------------------------------------------

@app.route('/learn/basic')
def basic_words():

    basic = ["hello", "sorry", "thank", "yes", "no"]

    cards = []

    for w in basic:
        cards.append({
            "word": w,
            "video": "/videos/" + w + ".mp4"
        })

    return render_template("words.html", words=cards)


# ---------------------------------------------------
# LEVEL 3 DAILY WORDS
# ---------------------------------------------------

@app.route('/learn/daily')
def daily_words():

    daily = ["water", "food", "help", "drink", "eat"]

    cards = []

    for w in daily:
        cards.append({
            "word": w,
            "video": "/videos/" + w + ".mp4"
        })

    return render_template("words.html", words=cards)


# ---------------------------------------------------
# LOAD WORD DATABASE
# ---------------------------------------------------

def load_words():

    words = []

    for file in os.listdir(VIDEO_FOLDER):

        if file.endswith(".mp4"):

            name = file.replace(".mp4", "")

            if len(name) > 1:
                words.append(name)

    words.sort()

    return words


# ---------------------------------------------------
# FLASHCARDS
# ---------------------------------------------------

@app.route('/flashcards')
def flashcards():

    words = load_words()

    cards = []

    for w in words:
        cards.append({
            "word": w,
            "video": "/videos/" + w + ".mp4"
        })

    return render_template("words.html", words=cards)


# ---------------------------------------------------
# QUIZ
# ---------------------------------------------------

@app.route('/quiz')
def quiz():

    words = load_words()

    correct = random.choice(words)

    options = random.sample(words, min(4, len(words)))

    if correct not in options:
        options[0] = correct

    random.shuffle(options)

    q = {
        "video": "/videos/" + correct + ".mp4",
        "options": options,
        "answer": correct
    }

    return render_template("quiz.html", q=q)


# ---------------------------------------------------
# LEADERBOARD
# ---------------------------------------------------

@app.route('/leaderboard')
def leaderboard():
    return render_template("leaderboard.html")


# ---------------------------------------------------
# TRANSLATOR PIPELINE
# ---------------------------------------------------

def processing(result):

    translator = Translator()

    trans_result = translator.translate(result, dest='en')

    pos_tag_result = pos_tagging(trans_result.text)

    filter_result = filter_words(pos_tag_result)

    sentence_reordering_result = sentence_reordering(filter_result)

    stop_word_eliminator_result = stop_word_eliminate(sentence_reordering_result)

    lemmatize_result = convert_lemma(stop_word_eliminator_result)

    video_result = video_conversion(lemmatize_result)

    return lemmatize_result, video_result


# ---------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------

def extract_features(sentence, index):

    return {

        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index < len(sentence)-1 else '',
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
    }


# ---------------------------------------------------
# POS TAGGING
# ---------------------------------------------------

def pos_tagging(result):

    ud_filename = 'ud_crf_postagger.sav'

    crf_from_pickle = pickle.load(open(ud_filename, 'rb'))

    sentences = [result]

    ud_sents = []

    for sent in sentences:

        features = [
            extract_features(sent.split(), idx)
            for idx in range(len(sent.split()))
        ]

        ud_results = crf_from_pickle.predict_single(features)

        ud_tups = [
            (sent.split()[idx], ud_results[idx])
            for idx in range(len(sent.split()))
        ]

        ud_sents.append(ud_tups)

    return ud_sents


# ---------------------------------------------------
# REMOVE PUNCTUATIONS
# ---------------------------------------------------

punctuations = [',', '?', '!', '.']


def removePunctuations(word):

    new_word = ""

    for ch in word:
        if ch not in punctuations:
            new_word += ch

    return new_word


def filter_words(ud_sents):

    new_sents = []

    for ud_tups in ud_sents:

        new_tups = []

        for tup in ud_tups:

            word = tup[0].lower()
            tag = tup[1]

            word = removePunctuations(word)

            new_tups.append((word, tag))

        new_sents.append(new_tups)

    return new_sents


# ---------------------------------------------------
# SENTENCE REORDERING
# ---------------------------------------------------

def sentence_reordering(ud_sents):

    reordered_sent_list = []

    for sent in ud_sents:

        reordered_sent = []
        verbs = []

        for tup in sent:

            if tup[1] == 'VERB':
                verbs.append(tup)
            else:
                reordered_sent.append(tup)

        reordered_sent = reordered_sent + verbs

        reordered_sent_list.append(reordered_sent)

    return reordered_sent_list


# ---------------------------------------------------
# STOP WORD REMOVAL
# ---------------------------------------------------

def stop_word_eliminate(reordered_sent_list):

    stop_words = ['to','is','the','a','an','and','are','was']

    isl_sent_list = []

    for reordered_sent in reordered_sent_list:

        isl_sent = []

        for tup in reordered_sent:

            if tup[0] not in stop_words:
                isl_sent.append(tup)

        isl_sent_list.append(isl_sent)

    return isl_sent_list


# ---------------------------------------------------
# LEMMATIZATION
# ---------------------------------------------------

def convert_lemma(isl_sent_list):

    lema_isl_sent_list = []

    for isl_sent in isl_sent_list:

        isl_sent_lem = []

        for word_tuple in isl_sent:
            isl_sent_lem.append(word_tuple[0])

        lema_isl_sent_list.append(isl_sent_lem)

    return lema_isl_sent_list


# ---------------------------------------------------
# VIDEO GENERATION
# ---------------------------------------------------

def video_conversion(lema_isl_sent_list):

    for isl_sent in lema_isl_sent_list:

        video_array = []

        for word_tuple in isl_sent:

            word_tuple = str(word_tuple).lower()

            word_path = VIDEO_FOLDER + "/" + word_tuple + ".mp4"

            if os.path.isfile(word_path):

                video_array.append(
                    VideoFileClip(word_path).resize((500,380))
                )

            else:

                for ch in word_tuple:

                    letter_path = VIDEO_FOLDER + "/" + ch.upper() + ".mp4"

                    if os.path.isfile(letter_path):

                        video_array.append(
                            VideoFileClip(letter_path).resize((500,380))
                        )

        sent = "".join(isl_sent) + ".mp4"

        final_clip = concatenate_videoclips(video_array, method='compose')

        final_clip.write_videofile("static/" + sent)

    return sent


# ---------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)