import datasets

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)
    wiki = datasets.load_dataset(
        './wikipedia.py',
        beam_runner='DirectRunner',
        language='ja',
        date='20210120')['train']

    print(len(wiki))
