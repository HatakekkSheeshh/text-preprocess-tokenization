def main():
    # EDA
    # wikitext-103
    from src.datasets.load_data import load
    load("wikitext-103")
    
if __name__ == "__main__":
    main()