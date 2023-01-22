import datetime as dt
import re

import datazimmer as dz
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from aswan import ParsedCollectionEvent
from nltk.sentiment import SentimentIntensityAnalyzer

from .collect import BBCPatientCollector, Collect, get_locales_table

plot_jpg = dz.ReportFile("sentiment.jpg")
latest_md = dz.ReportFile("latest.md")


class Article(dz.AbstractEntity):

    aid = dz.Index & str
    published = dt.datetime
    url = str
    title = str
    summary = str


renamer = {
    "lastPublished": Article.published,
    "assetId": Article.aid,
}


@dz.register(outputs_nocache=[plot_jpg], dependencies=[Collect])
def step(last_days: int):
    c = Collect()
    coll_df = pd.concat(map(_parse_pcev, c.get_all_events(BBCPatientCollector))).rename(
        columns=renamer
    )
    ua_df = coll_df.drop_duplicates(subset=Article.aid).pipe(dz.parse_df, Article)
    locale_base = get_locales_table().dropna()

    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    score_bases = [Article.summary, Article.title]

    score_dfs = {
        k: pd.DataFrame(ua_df[k].apply(sia.polarity_scores).tolist(), index=ua_df.index)
        for k in score_bases
    }
    comp_score_df = pd.DataFrame(
        {f"{b}_compound_score": score_dfs[b].loc[:, "compound"] for b in score_bases}
    ).assign(summary_minus_title=lambda df: df.diff(axis=1).iloc[:, -1])

    scored_regions = (
        coll_df.loc[:, [Article.aid, "pid"]]
        .drop_duplicates()
        .merge(locale_base.assign(pid=lambda df: df.iloc[:, -1].str.strip().tolist()))
        .merge(
            comp_score_df.assign(
                **{Article.published: ua_df[Article.published]}
            ).reset_index()
        )
    )

    score_cols = comp_score_df.columns.to_list()
    score_pivots = (
        scored_regions.pivot_table(
            index=Article.published, columns="region", values=score_cols
        )
        .rolling("21D")
        .mean()
        .resample("2D")
        .last()
        .tail(last_days // 2)
    )

    fig, axes = plt.subplots(
        nrows=len(score_cols), ncols=1, sharex=True, figsize=(13, 20)
    )

    for c, ax in zip(score_cols, axes):
        score_pivots.loc[:, c].plot(
            ax=ax, legend=False, title=" ".join(c.title().split("_"))
        )

    handles, labels = ax.get_legend_handles_labels()
    _ = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=5,
        fancybox=True,
        shadow=True,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(plot_jpg.current_path.as_posix())

    (
        scored_regions.sort_values(Article.published)
        .groupby("region")
        .last()
        .reset_index()
        .merge(ua_df.reset_index(), how="left")
        .loc[:, ["region", *ua_df.columns, *score_cols]]
        .to_markdown(latest_md.current_path, index=False)
    )


def _parse_pcev(pcev: ParsedCollectionEvent):
    push_id, page_num = re.findall("(uk-england-\d+).*pageNumber%2F(\d+)", pcev.url)[0]
    return pd.DataFrame(pcev.content["payload"][0]["body"]["results"]).assign(
        page=int(page_num), pid=push_id
    )
