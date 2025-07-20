use std::env;
use std::process;

use rust_test::sdf;
use rust_test::svosdf;

use sdf::*;
use svosdf::*;

pub struct Params {
    pub file_in: String,
    pub file_out: String,
    pub brick_size: u32,
    pub max_depth: u32,
    pub threshold: f32,
}

fn parse_args(args: &[String]) -> Result<Params, &str> {
    if args.len() < 3 {
        return Err("Not enough arguments");
    }

    let file_in = args[1].clone();
    let file_out = args[2].clone();

    let mut brick_size = 8;
    let mut max_depth = 8;
    let mut threshold = 0.01;

    let mut i = 3;
    while i < args.len() {
        match &args[i][..] {
            "-b" | "--brick-size" => {
                if i + 1 < args.len() {
                    brick_size = args[i + 1].parse().unwrap_or(8);
                    i += 2;
                } else {
                    return Err("Missing brick size value");
                }
            }
            "-d" | "--max-depth" => {
                if i + 1 < args.len() {
                    max_depth = args[i + 1].parse().unwrap_or(8);
                    i += 2;
                } else {
                    return Err("Missing max depth value");
                }
            }
            "-t" | "--threshold" => {
                if i + 1 < args.len() {
                    threshold = args[i + 1].parse().unwrap_or(0.01);
                    i += 2;
                } else {
                    return Err("Missing threshold value");
                }
            }
            _ => i += 1,
        }
    }

    Ok(Params {
        file_in,
       file_out,
       brick_size,
       max_depth,
       threshold,
    })
}

fn print_usage() {
    println!("Usage: svosdf input.sdf output.svosdf [options]");
    println!("Options:");
    println!("  -b, --brick-size <size>    Brick size (default: 8)");
    println!("  -d, --max-depth <depth>    Maximum octree depth (default: 8)");
    println!("  -t, --threshold <value>    Distance threshold for subdivision (default: 0.01)");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let params = parse_args(&args).unwrap_or_else(|err| {
        println!("Argument error: {}", err);
        print_usage();
        process::exit(1);
    });

    println!("Loading SDF: {}", params.file_in);
    let sdf = load_sdf_zlib(&params.file_in).expect("SDF loading failed");

    println!("Building sparse voxel octree...");
    println!("  Brick size: {}", params.brick_size);
    println!("  Max depth: {}", params.max_depth);
    println!("  Threshold: {}", params.threshold);

    let svo_sdf = SvoSdf::from_sdf(&sdf, params.brick_size, params.max_depth, params.threshold);

    let original_size = sdf.voxels.len() * std::mem::size_of::<u16>();
    let compressed_size = svo_sdf.calculate_memory_usage();
    let compression_ratio = (original_size as f32 / compressed_size as f32) * 100.0;

    println!("Compression results:");
    println!("  Original size: {} bytes", original_size);
    println!("  Compressed size: {} bytes", compressed_size);
    println!("  Compression ratio: {:.1}%", compression_ratio);
    println!("  Memory reduction: {:.1}%", 100.0 - (compressed_size as f32 / original_size as f32) * 100.0);

    println!("Saving sparse voxel octree: {}", params.file_out);
    svo_sdf.save(&params.file_out).expect("Failed to save SVO SDF");

    println!("Done!");
}
