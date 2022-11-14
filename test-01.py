"""
    @author Richard Álvarez
    @date Sat Nov 5 2022
    @desc Simple file cleaning and VADER sentiment analysis on a text file. Outputs a very simple graph.
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysbd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from cleantext import clean
from tqdm import tqdm

def main(argv: list[str] | None = None) -> None:   

    # Globals
    DATA = None
    TITLE = argv[1] or argv[0] or 'NULL'
    FILENAME = ''
    SID = SentimentIntensityAnalyzer()
    SEG = pysbd.Segmenter(language="en", clean=True)
    
    sentiment_compound_ls = []
    sentiment_score_ls = []
    sentence_ls = []

    print("Reading file....")

    with open(argv[0], "r+") as file:
            raw = file.read()
            # Simple cleaning
            DATA = raw.replace("\n\r", " ")
            DATA = raw.replace("\n", " ")
            DATA = raw.replace("\r", " ")
            DATA = " ".join(raw.split())
            
    print("Cleaning text....")

    novel_clean_str = clean(DATA,
        fix_unicode=True,             
        to_ascii=True,                
        lower=True,                   
        no_line_breaks=False,         
        no_urls=False,                
        no_emails=False,              
        no_phone_numbers=False,       
        no_numbers=False,             
        no_digits=False,              
        no_currency_symbols=False,    
        no_punct=False,               
        lang="en"                     
    )

   
    DATA = SEG.segment(DATA) 
    
    DATA = [segment.strip() for segment in DATA]

    print("\nFinished loading '", TITLE, "'\n", sep='')

    print("applying VADER sentiment analysis on each sentence....")

    VADER_sentence_pbar = tqdm(DATA, unit="sentences", colour="red")

    for sentence in VADER_sentence_pbar:
        sentiment_compound_ls.append(SID.polarity_scores(sentence)['compound'])
        sentiment_score_ls.append(SID.polarity_scores(sentence))
        sentence_ls.append(sentence)

    sentiment_df = pd.DataFrame({"vader": sentiment_compound_ls, "sentence": sentence_ls}) 
    now = datetime.now()

    FILENAME = ''.join(["/sentiment_", TITLE,  "_", now.strftime("%H-%M-%b-%d-%Y")])
   
    print("writing out each sentence sentiment results....")
    with open("res/{filename}.txt".format(filename=FILENAME), "a") as write_file:
        for i in tqdm(range(len(sentiment_score_ls)), unit="sentences", colour="red"):
            write_file.write(sentence_ls[i])
            write_file.write("\n")
            for k in sorted(sentiment_score_ls[i], reverse=True):
                write_file.write("{0}: {1}, ".format(k, sentiment_score_ls[i][k], end=""))
            write_file.write("\n")

    win_per = .10
    win_size = int(sentiment_df.shape[0] * win_per)
    plot = sentiment_df["vader"].rolling(window=win_size, center=True).mean().plot(
        title=TITLE
    )
    plot.set_xlabel("Sentence")
    plot.set_ylabel("Compound valence score (rolling)")
    plot.get_figure().savefig("fig/{filename}.pdf".format(filename=FILENAME))


if __name__ == "__main__":
    main(sys.argv[1:])