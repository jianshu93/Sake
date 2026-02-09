// src/main.rs
//
// DiskANN over MinHash sketches (u16) using anndists::dist::DistHamming,
// reporting raw collision probability / "raw jaccard" as 1 - normalized_hamming.
//
// Files per prefix:
//   <prefix>.diskann       : DiskANN index file (vectors are u16)
//   <prefix>.genomes.txt   : genome file paths in EXACT vector-id order (0..n-1)
//   <prefix>.idmap.tsv     : TSV mapping vector_id -> genome path (same order)
//   <prefix>.params.json   : parameters used to build/sketch
//
// Commands:
//   todiskann  : build index from a genome list
//   search     : search query genomes against an existing index

use clap::{Arg, ArgAction, Command};
use needletail::{parse_fastx_file, Sequence};
use num;
use num_traits::{NumCast, PrimInt, ToPrimitive};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::sync::Once;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use kmerutils::base::{
    alphabet::Alphabet2b,
    kmergenerator::*,
    sequence::Sequence as SequenceStruct,
    CompressedKmerT, Kmer16b32bit, Kmer32bit, Kmer64bit, KmerBuilder,
};
use kmerutils::sketcharg::{DataType, SeqSketcherParams, SketchAlgo};
use kmerutils::sketching::setsketchert::{OptDensHashSketch, RevOptDensHashSketch, SeqSketcherT};

use anndists::dist::DistHamming;
use rust_diskann::{DiskANN, DiskAnnParams};

static INIT_RAYON: Once = Once::new();

fn init_rayon_global(threads: usize) {
    INIT_RAYON.call_once(|| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("failed to init global rayon threadpool");
    });
}

/// Converts ASCII-encoded bases (from Needletail) into our `SequenceStruct`.
fn ascii_to_seq(bases: &[u8]) -> Result<SequenceStruct, ()> {
    let alphabet = Alphabet2b::new();
    let mut seq = SequenceStruct::with_capacity(2, bases.len());
    seq.encode_and_add(bases, &alphabet);
    Ok(seq)
}

/// Read lines (one path per line) from a text file.
fn read_list_file(filepath: &str) -> Vec<String> {
    let file =
        File::open(filepath).unwrap_or_else(|e| panic!("Cannot open list file {filepath}: {e}"));
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| line.expect("Error reading list file"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Write genome list file in exact order.
fn write_genome_list(filepath: &str, genomes: &[String]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(filepath)?);
    for g in genomes {
        writeln!(w, "{g}")?;
    }
    Ok(())
}

/// Read genome list file in exact order.
fn read_genome_list(filepath: &str) -> io::Result<Vec<String>> {
    Ok(read_list_file(filepath))
}

/// Write idmap TSV: `vector_id<TAB>genome_path`
fn write_idmap_tsv(filepath: &str, genomes: &[String]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(filepath)?);
    for (i, g) in genomes.iter().enumerate() {
        writeln!(w, "{}\t{}", i, g)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixParams {
    // sketch params
    kmer_size: usize,
    sketch_size: usize,
    densification: usize, // 0 optdens, 1 revoptdens
    threads: usize,

    // diskann params
    max_degree: usize,
    build_beam_width: usize,
    alpha: f32,

    // type info (we fix u16)
    sketch_elem_size: u8,
    sketch_elem_type: String,
    distance: String,

    // hash seed (must match between build/search)
    hash_seed: u64,
}

fn params_path(prefix: &str) -> String {
    format!("{prefix}.params.json")
}
fn index_path(prefix: &str) -> String {
    format!("{prefix}.diskann")
}
fn genomes_path(prefix: &str) -> String {
    format!("{prefix}.genomes.txt")
}
fn idmap_path(prefix: &str) -> String {
    format!("{prefix}.idmap.tsv")
}

fn save_params(prefix: &str, p: &PrefixParams) -> io::Result<()> {
    let s = serde_json::to_string_pretty(p).unwrap();
    fs::write(params_path(prefix), s)
}

fn load_params(prefix: &str) -> io::Result<PrefixParams> {
    let s = fs::read_to_string(params_path(prefix))?;
    Ok(serde_json::from_str(&s).unwrap())
}

/// Strict, order-preserving parallel sketching:
/// returns Vec<Vec<u16>> aligned to `file_paths` order.
///
/// IMPORTANT: we REQUIRE Sketcher::Sig = u16 at compile time.
fn sketch_files_ordered_u16<Kmer, Sketcher, F>(
    file_paths: &[String],
    sketcher: &Sketcher,
    kmer_hash_fn: F,
) -> Vec<Vec<u16>>
where
    Kmer: CompressedKmerT + KmerBuilder<Kmer> + Send + Sync,
    <Kmer as CompressedKmerT>::Val: num::PrimInt + Send + Sync + std::fmt::Debug,
    KmerGenerator<Kmer>: KmerGenerationPattern<Kmer>,

    Sketcher: SeqSketcherT<Kmer, Sig = u16> + Sync,

    F: Fn(&Kmer) -> <Kmer as CompressedKmerT>::Val + Send + Sync + Copy,
{
    // We keep a local Vec<(original_index, signature)> and then sort by index
    // to guarantee that final vectors follow file_paths order exactly.
    let mut indexed: Vec<(usize, Vec<u16>)> = file_paths
        .par_iter()
        .enumerate()
        .map(|(i, path)| {
            let mut sequences: Vec<SequenceStruct> = Vec::new();
            let mut reader =
                parse_fastx_file(path).unwrap_or_else(|e| panic!("Invalid FASTA/Q {path}: {e}"));

            while let Some(record) = reader.next() {
                let rec = record.unwrap_or_else(|e| panic!("Error reading record in {path}: {e}"));
                let seq_norm = rec.normalize(false).into_owned();
                let seq = ascii_to_seq(&seq_norm).unwrap();
                sequences.push(seq);
            }

            let sequences_ref: Vec<&SequenceStruct> = sequences.iter().collect();

            // returns Vec<Vec<u16>>, and for "seqs" interface inner vec has size 1
            let signature = sketcher.sketch_compressedkmer_seqs(&sequences_ref, kmer_hash_fn);
            let sig_u16 = signature
                .get(0)
                .cloned()
                .unwrap_or_else(|| panic!("sketcher returned empty signature for {path}"));

            (i, sig_u16)
        })
        .collect();

    indexed.sort_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, v)| v).collect()
}

/// Canonicalize (min(kmer, revcomp)) -> pack -> XXH3 -> return Kmer::Val (u32/u64).
/// Safe for k<=32 (never shifts by 64).
pub fn make_xxh3_canonical_kmer_hash_fn<Kmer>(
    seed: u64,
) -> impl Fn(&Kmer) -> Kmer::Val + Copy + Send + Sync
where
    Kmer: CompressedKmerT + KmerBuilder<Kmer> + Copy,
    Kmer::Val: PrimInt + ToPrimitive + NumCast,
{
    move |kmer: &Kmer| -> Kmer::Val {
        // canonicalize
        let rc = kmer.reverse_complement();
        let canonical = if rc < *kmer { rc } else { *kmer };

        // DNA packing: 2 bits per base
        let k: usize = canonical.get_nb_base() as usize;
        let bits: usize = 2usize * k;

        // safe mask (avoid 1<<64)
        let mask_u64: u64 = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };

        // convert packed bits to u64 safely (works for u32/u64)
        let packed_u64: u64 = canonical
            .get_compressed_value()
            .to_u64()
            .expect("Kmer::Val must fit in u64")
            & mask_u64;

        // hash packed bytes
        let h64: u64 = xxh3_64_with_seed(&packed_u64.to_le_bytes(), seed);

        // output type:
        // - if Val is u32-ish => use low 32 bits
        // - if Val is u64-ish => keep full 64
        if std::mem::size_of::<Kmer::Val>() <= 4 {
            let low32 = h64 as u32 as u64; // intentional truncation
            NumCast::from(low32).expect("casting hash->Val failed")
        } else {
            NumCast::from(h64).expect("casting hash->Val failed")
        }
    }
}

/// Dispatch sketching by k-mer size.
/// Fixed Sig=u16 for the sketcher output.
fn sketch_with_kmer_dispatch_u16(
    paths: &[String],
    kmer_size: usize,
    sketch_size: usize,
    densification: usize, // 0 optdens, 1 revoptdens
    hash_seed: u64,
) -> Vec<Vec<u16>> {
    let sketch_args = SeqSketcherParams::new(
        kmer_size,
        sketch_size,
        SketchAlgo::OPTDENS, // label; actual is controlled by sketcher type
        DataType::DNA,
    );

    if kmer_size <= 14 {
        let kmer_hash_fn = make_xxh3_canonical_kmer_hash_fn::<Kmer32bit>(hash_seed);

        match densification {
            0 => {
                let sketcher = OptDensHashSketch::<Kmer32bit, f32>::new(&sketch_args);
                sketch_files_ordered_u16::<Kmer32bit, _, _>(paths, &sketcher, kmer_hash_fn)
            }
            1 => {
                let sketcher = RevOptDensHashSketch::<Kmer32bit, f32>::new(&sketch_args);
                sketch_files_ordered_u16::<Kmer32bit, _, _>(paths, &sketcher, kmer_hash_fn)
            }
            _ => panic!("densification must be 0 or 1"),
        }
    } else if kmer_size == 16 {
        let kmer_hash_fn = make_xxh3_canonical_kmer_hash_fn::<Kmer16b32bit>(hash_seed);

        match densification {
            0 => {
                let sketcher = OptDensHashSketch::<Kmer16b32bit, f32>::new(&sketch_args);
                sketch_files_ordered_u16::<Kmer16b32bit, _, _>(paths, &sketcher, kmer_hash_fn)
            }
            1 => {
                let sketcher = RevOptDensHashSketch::<Kmer16b32bit, f32>::new(&sketch_args);
                sketch_files_ordered_u16::<Kmer16b32bit, _, _>(paths, &sketcher, kmer_hash_fn)
            }
            _ => panic!("densification must be 0 or 1"),
        }
    } else if kmer_size <= 32 {
        let kmer_hash_fn = make_xxh3_canonical_kmer_hash_fn::<Kmer64bit>(hash_seed);

        match densification {
            0 => {
                let sketcher = OptDensHashSketch::<Kmer64bit, f32>::new(&sketch_args);
                sketch_files_ordered_u16::<Kmer64bit, _, _>(paths, &sketcher, kmer_hash_fn)
            }
            1 => {
                let sketcher = RevOptDensHashSketch::<Kmer64bit, f32>::new(&sketch_args);
                sketch_files_ordered_u16::<Kmer64bit, _, _>(paths, &sketcher, kmer_hash_fn)
            }
            _ => panic!("densification must be 0 or 1"),
        }
    } else {
        panic!("kmer_size cannot exceed 32 and must not be 15!");
    }
}

/// Write search results as TSV.
/// `hits`: Vec[(query_index, [(vector_id, hamming_dist)])]
fn write_search_results(
    mut out: Box<dyn Write>,
    query_paths: &[String],
    hits: &[(usize, Vec<(u32, f32)>)],
    ref_genomes: &[String],
) -> io::Result<()> {
    writeln!(out, "Query\tHit\tVectorID\tRawJaccard\tHammingDist")?;
    for (qi, nn) in hits {
        let q = &query_paths[*qi];
        for (id, dist) in nn {
            let hit = &ref_genomes[*id as usize];
            let raw_j = (1.0 - *dist).clamp(0.0, 1.0);
            writeln!(
                out,
                "{}\t{}\t{}\t{:.6}\t{:.6}",
                q, hit, id, raw_j, dist
            )?;
        }
    }
    Ok(())
}

/// Small internal sanity check: take a few reference vectors and query them
/// back against the index. This is just to confirm DiskANN's ID mapping
/// is consistent with our 0..n-1 order. No CLI flag, just stderr logs.
fn sanity_check_index_mapping(
    idx: &DiskANN<u16, DistHamming>,
    vectors: &[Vec<u16>],
    num_checks: usize,
    build_beam_width: usize,
) {
    let n = vectors.len();
    if n == 0 {
        eprintln!("SANITY: index has 0 vectors, skipping sanity check.");
        return;
    }
    let checks = n.min(num_checks.max(1));
    let beam = build_beam_width.max(64);

    eprintln!(
        "SANITY: checking mapping on {} / {} reference vectors (beam_width={})",
        checks, n, beam
    );
    for i in 0..checks {
        let nn = idx.search_with_dists(&vectors[i], 1, beam);
        if let Some((id, dist)) = nn.first().copied() {
            eprintln!(
                "SANITY: ref vec {} -> nearest id {} (dist={:.6})",
                i, id, dist
            );
        } else {
            eprintln!("SANITY: ref vec {} -> search returned no neighbors", i);
        }
    }
}

fn main() {
    println!("\n ************** initializing logger *****************\n");
    let _ = env_logger::Builder::from_default_env().try_init();

    let matches = Command::new("sake")
        .version("0.1.0")
        .about("lightning-fast and space-efficient genome search index based on DiskANN and b-bit One Permutation MinHash")
        .subcommand_required(true)
        .subcommand(
            Command::new("todiskann")
                .about("Build DiskANN index from genome sketches")
                .arg(
                    Arg::new("prefix")
                        .long("prefix")
                        .short('p')
                        .required(true)
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("reference_list")
                        .long("reference_list")
                        .short('r')
                        .required(true)
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("kmer_size")
                        .long("kmer_size")
                        .short('k')
                        .default_value("16")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("sketch_size")
                        .long("sketch_size")
                        .short('s')
                        .default_value("8192")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("densification")
                        .long("densification")
                        .short('d')
                        .help("0 = optimal densification, 1 = Reverse Optimal Densification")
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("threads")
                        .long("threads")
                        .short('t')
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("max_degree")
                        .long("max_degree")
                        .default_value("256")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("build_beam_width")
                        .long("build_beam_width")
                        // bump a bit vs 128 for high-dim u16 sketches
                        .default_value("512")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("alpha")
                        .long("alpha")
                        .default_value("1.8")
                        .value_parser(clap::value_parser!(f32))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("hash_seed")
                        .long("hash_seed")
                        .help("XXH3 seed for canonical k-mer hashing (MUST match between build & search)")
                        .default_value("1337")
                        .value_parser(clap::value_parser!(u64))
                        .action(ArgAction::Set),
                ),
        )
        .subcommand(
            Command::new("search")
                .about("Search query genomes against an existing DiskANN index")
                .arg(
                    Arg::new("prefix")
                        .long("prefix")
                        .short('p')
                        .required(true)
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("query_list")
                        .long("query_list")
                        .short('q')
                        .required(true)
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("k")
                        .long("k")
                        .help("Top-k neighbors to return")
                        .default_value("10")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("beam_width")
                        .long("beam_width")
                        .help("Search beam width")
                        // default reasonably large; override if needed
                        .default_value("512")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("threads")
                        .long("threads")
                        .short('t')
                        .help("Threads for sketching/search")
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize))
                        .action(ArgAction::Set),
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .short('o')
                        .help("Output TSV (default stdout)")
                        .required(false)
                        .action(ArgAction::Set),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("todiskann", m)) => {
            let prefix = m.get_one::<String>("prefix").unwrap().to_string();
            let reference_list = m.get_one::<String>("reference_list").unwrap().to_string();

            let kmer_size = *m.get_one::<usize>("kmer_size").unwrap();
            let sketch_size = *m.get_one::<usize>("sketch_size").unwrap();
            let dens = *m.get_one::<usize>("densification").unwrap();
            let threads = *m.get_one::<usize>("threads").unwrap();

            let max_degree = *m.get_one::<usize>("max_degree").unwrap();
            let build_beam_width = *m.get_one::<usize>("build_beam_width").unwrap();
            let alpha = *m.get_one::<f32>("alpha").unwrap();

            let hash_seed = *m.get_one::<u64>("hash_seed").unwrap();

            init_rayon_global(threads);

            let ref_genomes = read_list_file(&reference_list);
            eprintln!(
                "Building index for {} reference genomes (k={}, sketch_size={}, dens={}, seed={})",
                ref_genomes.len(),
                kmer_size,
                sketch_size,
                dens,
                hash_seed
            );

            // Save genome list in the exact input order (this defines vector IDs)
            let genomes_file = genomes_path(&prefix);
            write_genome_list(&genomes_file, &ref_genomes).expect("failed writing genomes list");

            // Sketch reference genomes (order preserved)
            let vectors_u16 = sketch_with_kmer_dispatch_u16(
                &ref_genomes,
                kmer_size,
                sketch_size,
                dens,
                hash_seed,
            );
            assert_eq!(
                vectors_u16.len(),
                ref_genomes.len(),
                "sketch count mismatch vs genomes list order"
            );
            if !vectors_u16.is_empty() {
                eprintln!(
                    "Sketched {} vectors, dim={} (u16)",
                    vectors_u16.len(),
                    vectors_u16[0].len()
                );
            }

            // Build DiskANN with DistHamming<u16>
            let idx_file = index_path(&prefix);
            let params = DiskAnnParams {
                max_degree,
                build_beam_width,
                alpha,
            };

            let idx = DiskANN::<u16, DistHamming>::build_index_with_params(
                &vectors_u16,
                DistHamming,
                &idx_file,
                params,
            )
            .expect("DiskANN build failed");

            eprintln!(
                "OK: built {} (num_vectors={}, dim={})",
                idx_file, idx.num_vectors, idx.dim
            );

            // Internal sanity check: do a few ref->ref queries to see mapping.
            sanity_check_index_mapping(&idx, &vectors_u16, 3, build_beam_width);

            // Write params json
            let pp = PrefixParams {
                kmer_size,
                sketch_size,
                densification: dens,
                threads,
                max_degree,
                build_beam_width,
                alpha,
                sketch_elem_size: 2,
                sketch_elem_type: "u16".to_string(),
                distance: "anndists::dist::DistHamming".to_string(),
                hash_seed,
            };
            save_params(&prefix, &pp).expect("failed writing params");

            // Write idmap.tsv using the same genome order
            let idmap_file = idmap_path(&prefix);
            write_idmap_tsv(&idmap_file, &ref_genomes).expect("failed writing idmap");

            eprintln!("OK: genomes order {}", genomes_file);
            eprintln!("OK: idmap {}", idmap_file);
            eprintln!("OK: params {}", params_path(&prefix));
        }

        Some(("search", m)) => {
            let prefix = m.get_one::<String>("prefix").unwrap().to_string();
            let query_list = m.get_one::<String>("query_list").unwrap().to_string();
            let k = *m.get_one::<usize>("k").unwrap();
            let beam_width = *m.get_one::<usize>("beam_width").unwrap();
            let threads = *m.get_one::<usize>("threads").unwrap();
            let output = m.get_one::<String>("output").cloned();

            init_rayon_global(threads);

            // Load params to ensure we sketch queries exactly the same way
            let pp = load_params(&prefix).expect("failed reading params");
            if pp.sketch_elem_type != "u16" || pp.sketch_elem_size != 2 {
                panic!("prefix params indicate non-u16 sketches, but this binary is fixed to u16");
            }

            // Load reference genome order (defines vector IDs)
            let ref_genomes =
                read_genome_list(&genomes_path(&prefix)).expect("failed reading genomes list");

            // Open index
            let idx =
                DiskANN::<u16, DistHamming>::open_index_with(&index_path(&prefix), DistHamming)
                    .expect("failed opening index");

            eprintln!(
                "Opened index: num_vectors={}, dim={}, metric=Hamming<u16>",
                idx.num_vectors, idx.dim
            );
            eprintln!(
                "Ref genomes: {}, queries_from_list: {:?}, k={}, beam_width={}, threads={}",
                ref_genomes.len(),
                query_list,
                k,
                beam_width,
                threads
            );

            if idx.num_vectors != ref_genomes.len() {
                eprintln!(
                    "WARNING: index num_vectors ({}) != ref_genomes.len() ({})",
                    idx.num_vectors,
                    ref_genomes.len()
                );
            }

            // Read query list
            let query_genomes = read_list_file(&query_list);
            eprintln!("Sketching {} query genomes…", query_genomes.len());

            // Sketch queries (order preserved; match params)
            let query_vectors_u16 = sketch_with_kmer_dispatch_u16(
                &query_genomes,
                pp.kmer_size,
                pp.sketch_size,
                pp.densification,
                pp.hash_seed,
            );
            if !query_vectors_u16.is_empty() {
                eprintln!(
                    "Query vector dim={} (must match index dim={})",
                    query_vectors_u16[0].len(),
                    idx.dim
                );
            }

            // Search each query
            let hits: Vec<(usize, Vec<(u32, f32)>)> = query_vectors_u16
                .par_iter()
                .enumerate()
                .map(|(qi, qv)| {
                    let nn = idx.search_with_dists(qv, k, beam_width);
                    (qi, nn)
                })
                .collect();

            // Output
            let out: Box<dyn Write> = match output {
                Some(path) => Box::new(BufWriter::new(
                    File::create(&path)
                        .unwrap_or_else(|e| panic!("cannot create output {path}: {e}")),
                )),
                None => Box::new(BufWriter::new(io::stdout())),
            };
            write_search_results(out, &query_genomes, &hits, &ref_genomes).expect("write failed");
        }

        _ => unreachable!(),
    }
}