import sys

def read_tsv(file_path):
    """Read a TSV file and return its contents as a list of lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def compare_tsv(file1, file2):
    """Compare two TSV files line by line."""
    content1 = read_tsv(file1)
    content2 = read_tsv(file2)

    if content1 == content2:
        print("The TSV files are identical.")
    else:
        print("The TSV files are different.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_tsv.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    compare_tsv(file1, file2)
