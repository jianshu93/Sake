# Granne: Graph-based Approximate Nearest Neighbor using On-disk indexes for Rapid and Scalable Genome Search
Granne (***GR***aph-based ***A***pproximate ***N***earest ***NE***ighbor, it means neighbor in Swedish) is a lightning-fast and space-efficient genome nearest neighbor search index based on DiskANN and b-bit One Permutation MinHash.



## Install
```bash
git clone https://github.com/jianshu93/Granne
cd Sake
cargo build --release
./target/release/granne -h
```

## usage
```bash
$ granne -h

 ************** initializing logger *****************

lightning-fast and space-efficient genome search index based on DiskANN and b-bit One Permutation MinHash

Usage: sake <COMMAND>

Commands:
  todiskann  Build DiskANN index from genomes, genomes will be sketched first
  search     Search query genomes against an existing DiskANN index, query genomes will be sketched first
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

```


## Reference
Paper to come


