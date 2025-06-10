

def main():
    print("Welcome to My Python Project!")
    loader = ParallelFileLoader("example.txt", chunk_size=1024)
    chunks = loader.load_in_chunks(num_threads=4)
    print(f"Loaded {len(chunks)} chunks.")

if __name__ == "__main__":
    main()