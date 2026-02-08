# Sake: Rapid and Scalable Genome Search with On-disk Indexes
Sake (pronouced as Sah-key) is a lightning-fast and space-efficient genome search index based on DiskANN and b-bit One Permutation MinHash. I coded most of it when drinking Sake.


## Install
```bash
git clone https://github.com/jianshu93/Sake
cd Sake
cargo build --release
./target/release/sake -h


```

## usage
```bash
lightning-fast and space-efficient genome search index based on DiskANN and b-bit One Permutation MinHash

Usage: sake <COMMAND>

Commands:
  todiskann  Build DiskANN index from genome sketches
  search     Search query genomes against an existing DiskANN index
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

```


## Reference
Paper to come


