import sys
from tools import estimatePrice


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_price.py <mileage>")
        return
    mileage = int(sys.argv[1])
    res = estimatePrice(mileage)
    print(res)
    return res


if __name__ == "__main__":
    main()
