import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("main.py")


if __name__ == "__main__":
    main()
