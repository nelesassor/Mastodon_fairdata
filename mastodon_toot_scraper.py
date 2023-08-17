import pandas as pd
import os
import requests
import json
path='../data'

# scrape tweets from mastodon

df = pd.DataFrame({
    'Mastodonservers': ['sciencemastodon.com', 'scicomm.xyz', 'astrodon.social', 'genomic.social',
              'mstdn.science', 'ecoevo.social', 'fediscience.org',
              'mastodon.green', 'mastodon.nz', 'med-mastodon.com',
              'toot.wales', 'chaos.social', 'newsie.social',
              'aus.social', 'mastodon.scot', 'mastodon.ie', 'qoto.org',
              'mastodon.au', 'home.social', 'mastodonapp.uk', 'fairpoints.social', 'NFDI.Social',
              'openbiblio.social'],
})

def search_hashtag_on_servers(df, hashtag, path):
    for server in df['Mastodonservers']:
        URL = f'https://{server}/api/v1/timelines/tag/{hashtag}'
        params = {
            'limit': 5000
        }

        results = []

        while True:
            r = requests.get(URL, params=params)
            r.raise_for_status()  # Throws an error if the request fails

            toots = r.json()

            if len(toots) == 0:
                break

            results.extend(toots)

            max_id = toots[-1]['id']
            params['max_id'] = max_id

        mastodon_df = pd.DataFrame(results)
        print(mastodon_df.shape)

        filename = f'{server.replace(".", "_")}_mastodon_toots_{hashtag}.csv'
        mastodon_df.to_csv(os.path.join(path, filename))


search_hashtag_on_servers(df, 'fairdata', '../data/mastodon/2906')


