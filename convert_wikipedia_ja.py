import datasets

wiki_ja = datasets.load_dataset(
    "./pretrain/wikipedia.py",
    beam_runner="DirectRunner",
    language="ja",
    date="20210120",
)["train"]
print(wiki_ja)

wiki_ja.save_to_disk("./wikipedia-ja")
