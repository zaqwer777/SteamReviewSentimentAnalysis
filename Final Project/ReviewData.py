import json

import numpy as np
import statistics
import pandas as pd
from textblob import TextBlob


def load_data(app_id):
    # Data folder
    data_path = "data/"

    json_filename = "review_" + app_id + ".json"

    data_filename = data_path + json_filename

    with open(data_filename, 'r', encoding="utf8") as in_json_file:
        review_data = json.load(in_json_file)

    return review_data


def describe_data(review_data):
    try:
        query_summary = review_data['query_summary']

        sentence = 'Number of reviews: {0} ({1} up ; {2} down)'
        sentence = sentence.format(query_summary["total_reviews"], query_summary["total_positive"],
                                   query_summary["total_negative"])
    except KeyError:
        query_summary = None

        sentence = 'Query summary cannot be found in the JSON file.'

    print(sentence)

    reviews = list(review_data['reviews'].values())

    sentence = 'Number of downloaded reviews: ' + str(len(reviews))
    print(sentence)

    return query_summary, reviews


def aggregate_reviews(app_id):
    review_data = load_data(app_id)

    (query_summary, reviews) = describe_data(review_data)

    # Ratio of reviews from query_summary
    ratio = query_summary['total_positive'] / query_summary['total_reviews']

    review_stats = dict()

    # Review ID
    review_stats['recommendationid'] = []

    # Meta-data regarding the reviewers
    review_stats['num_games_owned'] = []
    review_stats['num_reviews'] = []
    review_stats['playtime_forever'] = []

    # Meta-data regarding the reviews themselves
    review_stats['language'] = []
    review_stats['voted_up'] = []
    review_stats['votes_up'] = []
    review_stats['votes_funny'] = []
    review_stats['weighted_vote_score'] = []
    review_stats['comment_count'] = []
    review_stats['steam_purchase'] = []
    review_stats['received_for_free'] = []

    # Sentiment analysis
    review_stats['polarity'] = []
    review_stats['subjectivity'] = []
    review_stats['correct_guesses'] = 0
    review_stats['incorrect_guesses'] = 0
    review_stats['no_polarity_upvoted'] = 0
    review_stats['no_polarity_downvoted'] = 0

    for review in reviews:
        review_content = review['review']

        # Review ID
        review_stats['recommendationid'].append(review["recommendationid"])

        # Metadata for the author of the review
        review_stats['num_games_owned'].append(review['author']['num_games_owned'])
        review_stats['num_reviews'].append(review['author']['num_reviews'])
        review_stats['playtime_forever'].append(review['author']['playtime_forever'])

        # Metadata for the actual review
        review_stats['language'].append(review['language'])
        review_stats['voted_up'].append(review['voted_up'])
        review_stats['votes_up'].append(review['votes_up'])
        review_stats['votes_funny'].append(review['votes_funny'])
        review_stats['weighted_vote_score'].append(review['weighted_vote_score'])
        review_stats['comment_count'].append(review['comment_count'])
        review_stats['steam_purchase'].append(review['steam_purchase'])
        review_stats['received_for_free'].append(review['received_for_free'])

        # Sentiment analysis
        blob = TextBlob(review_content)
        review_stats['polarity'].append(blob.sentiment.polarity)
        review_stats['subjectivity'].append(blob.sentiment.subjectivity)

        # Filter the results based on voted_up and polarity
        if review['voted_up'] and blob.sentiment.polarity > 0.0:
            #print(review['voted_up'], blob.sentiment.polarity, review_stats['correct_guesses'])
            review_stats['correct_guesses'] += 1
        elif review['voted_up'] is False and blob.sentiment.polarity < 0.0:
            review_stats['correct_guesses'] += 1
        else:
            review_stats['incorrect_guesses'] += 1

        if blob.sentiment.polarity == 0.0:
            if review['voted_up'] is False:
                review_stats['no_polarity_downvoted'] += 1
            if review['voted_up'] is True:
                review_stats['no_polarity_upvoted'] += 1

    return review_stats, ratio


def aggregate_reviews_to_pandas(app_id):
    review_stats, ratio = aggregate_reviews(app_id)

    df = pd.DataFrame(data=review_stats)

    if "comment_count" in df.columns:
        df["comment_count"] = df["comment_count"].astype('int')
    if "weighted_vote_score" in df.columns:
        df["weighted_vote_score"] = df["weighted_vote_score"].astype('float')

    return df, ratio


def extract_reviews_for_language(df, top_languages, verbose=True):
    s = pd.Series([lang in top_languages for lang in df["language"]], name='language')
    df_extracted = df[s.values]

    return df_extracted


def analyze_app_id(app_id, languages_to_extract=None, including_all=False):
    df, ratio = aggregate_reviews_to_pandas(app_id)
    language = languages_to_extract
    df_extracted = extract_reviews_for_language(df, language)

    # Average values with 0.0 polarity results included
    if including_all:
        average_polarity = statistics.mean(df_extracted['polarity'])
        average_subjectivity = statistics.mean(df_extracted['subjectivity'])

        print(average_polarity, average_subjectivity)

    # Average values with 0.0 polarity results removed
    df_removed_polarity = df_extracted[df_extracted['polarity'] != 0.0]
    average_polarity = statistics.mean(df_removed_polarity['polarity'])

    average_subjectivity = statistics.mean(df_removed_polarity['subjectivity'])

    correct = statistics.mean(df_removed_polarity['correct_guesses'])
    incorrect = statistics.mean(df_removed_polarity['incorrect_guesses'])
    no_polarity_upvoted = statistics.mean(df_removed_polarity['no_polarity_upvoted'])
    no_polarity_downvoted = statistics.mean(df_removed_polarity['no_polarity_downvoted'])
    incorrect_without_no_polarity = incorrect - no_polarity_upvoted - no_polarity_downvoted

    print("Average Polarity of Game: ", average_polarity)
    print("Average Subjectivity of Game: ", average_subjectivity)
    print("Total Correct Guesses: ", correct)
    print("Total Incorrect Guesses: ", incorrect)
    print("Total Zero Polarity Upvoted: ", no_polarity_upvoted)
    print("Total Zero Polarity Downvoted: ", no_polarity_downvoted)
    print("Incorrect Guesses Without Zero Polarities: ", incorrect_without_no_polarity)
    print("Ratio of Polarity and Review Guess: ", correct / (correct + incorrect_without_no_polarity))
    print("Actual Ratio of Reviews: ", ratio)

    return df_extracted


def analyze_app_id_in_english(app_id):
    df = analyze_app_id(app_id, ['english'])
    return df


def main():
    app_id = "578080"

    # Analyze one appID
    analyze_app_id_in_english(app_id)

    return True


if __name__ == "__main__":
    main()
