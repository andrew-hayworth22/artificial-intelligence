import sys
from hmm import HMM

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        exit(1)

    config_path = sys.argv[1]
    hmm = HMM(config_path)

    prompt = "Enter sequence or 'exit' to exit: "
    sequence = input(prompt)
    while str.lower(sequence) != "exit":
        hmm.predict(sequence)
        sequence = input(prompt)
