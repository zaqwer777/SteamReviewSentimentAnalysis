import steamreviews


def main():
    from appids import appids

    # All the references
    request_params = dict()
    # Reference: https://partner.steamgames.com/doc/store/getreviews
    request_params['filter'] = 'all'  # reviews are sorted by helpfulness instead of chronology
    request_params['day_range'] = '28'  # focus on reviews which were published during the past four weeks
    steamreviews.download_reviews_for_app_id_batch(appids, chosen_request_params=request_params)

    return True


if __name__ == "__main__":
    main()
