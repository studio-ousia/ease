from sentsplitter import MosesSentenceSplitter




def main():
    print("a")
    split_sents = MosesSentenceSplitter('en')
    print("b")
    print(split_sents(['Hello World! Hello', 'again.']))

    

if __name__ == "__main__":
    main()