
def main():
    with open('./trivial_test.txt', 'r') as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    print(content)


if __name__ == '__main__':
    main()