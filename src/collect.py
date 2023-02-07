from itertools import product

import datazimmer as dz
import pandas as pd
from aswan import RequestJsonHandler, add_url_params

PAGES_TO_PULL = 4


class BBCPatientCollector(RequestJsonHandler):

    max_in_parallel = 4

    def get_sleep_time(self):
        return 2.5


class Collect(dz.DzAswan):
    name = "bbc-local"
    cron = "10 9 * * *"

    def prepare_run(self):

        push_ids = get_locales_table().iloc[:, -1].dropna().str.strip()
        argprod = product(list(range(1, PAGES_TO_PULL + 1)), push_ids)
        all_push_urls = list(map(create_push_url, argprod))
        self.starters = {BBCPatientCollector: all_push_urls}


def create_push_url(args):
    pnum, pager_id = args
    is_uk = "false"
    limit = 20
    push_base = "https://push.api.bbci.co.uk/batch"
    push_t = (
        f"/data/bbc-morph-lx-commentary-data-paged/assetUri/news%2Flive%2F{pager_id}"
        f"/isUk/{is_uk}/limit/{limit}/nitroKey/lx-nitro/pageNumber/{pnum}/version/1.5.6?timeout=5"
    )
    return add_url_params(push_base, {"t": push_t})


def get_locales_table():
    return (
        pd.read_table("locales.md", sep="|", header=0, skipinitialspace=True)
        .dropna(axis=1, how="all")
        .iloc[1:, :]
        .rename(columns=str.strip)
    )
