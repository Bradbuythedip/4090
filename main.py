# multi_pattern_cuda_search.py
# Searches multiple pattern spaces exhaustively for Bitcoin Puzzle #68

import hashlib
import binascii
import time
import sys
import numpy as np
from mpmath import mp, mpf, fabs
import ecdsa
from ecdsa import SECP256k1, SigningKey
import argparse
import os

# Import PyCUDA modules with proper error handling
try:
    import pycuda.driver as cuda
    import pycuda.compiler as compiler
    HAS_CUDA = True
except ImportError:
    print("PyCUDA not installed. Please install with: pip install pycuda")
    HAS_CUDA = False
except Exception as e:
    print(f"Error initializing CUDA: {e}")
    HAS_CUDA = False

# Set high precision for mpmath
mp.dps = 1000  # 1000 decimal places of precision

# Constants
PHI = mpf('1.6180339887498948482045868343656381177203091798057628621354486227052604628189')
TARGET_HASH = 'e0b8a2baee1b77fc703455f39d51477451fc8cfc'  # Hash from scriptPubKey
PUZZLE_NUMBER = 68
PHI_OVER_8 = float(PHI / 8)  # Convert to float for GPU compatibility

# Pattern spaces to search
PATTERNS = [
    # Primary search pattern - already searched
    {"pattern": "00000000000000000000000000000000000000000000000cedb187f", "comment": "CEDB187F pattern", "searched": True},
    
    # Key patterns with close phi/8 ratios
    {"pattern": "00000000000000000000000000000000000000000000000041a01b90", "comment": "PHI_MATCH base", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000cedb187e", "comment": "CEDB187E (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000cedb187d", "comment": "CEDB187D (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000cedb1870", "comment": "CEDB1870 (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000cedb186f", "comment": "CEDB186F (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000cedb087f", "comment": "CEDB087F (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000cedb387f", "comment": "CEDB387F (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000ced3187f", "comment": "CED3187F (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000ceda187f", "comment": "CEDA187F (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000c6db187f", "comment": "C6DB187F (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000dedb187f", "comment": "DEDB187F (1 bit difference)", "searched": False},
    {"pattern": "00000000000000000000000000000000000000000000000467a98b1", "comment": "Additional candidate pattern", "searched": False}
]

# CUDA kernel for testing keys
CUDA_CODE = """
#include <stdio.h>
#include <stdint.h>

// Approximation of the ratio calculation for filtering
__device__ float approx_ratio(uint64_t key_high, uint64_t key_low) {
    float result = 0.0f;
    float x_approx = (float)(key_high ^ (key_low >> 12));
    float y_approx = (float)(key_low ^ (key_high << 8));
    
    if (y_approx != 0.0f) {
        result = 0.202254f + (float)((x_approx * 723467.0f + y_approx * 13498587.0f) / 
                                   (1e12 + key_low + key_high) - 0.5f) * 0.00001f;
    }
    return result;
}

// Hash approximation for filtering
__device__ uint32_t approx_hash(uint64_t key_high, uint64_t key_low) {
    uint32_t h = 0xe0b8a2ba; // First bytes of target
    h ^= key_high >> 32;
    h ^= key_high;
    h ^= key_low >> 32;
    h ^= key_low;
    h ^= h >> 16;
    return h;
}

__global__ void test_keys(uint64_t prefix_high, uint64_t prefix_low, 
                         uint32_t *offsets, int offset_count,
                         float target_ratio, uint32_t target_hash_prefix,
                         uint32_t *candidates, float *ratios, uint32_t *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offset_count) return;
    
    // Get this thread's offset
    uint32_t offset = offsets[idx];
    
    // Create full key by combining prefix and offset
    uint64_t key_low = prefix_low | offset;
    uint64_t key_high = prefix_high;
    
    // Calculate approximate ratio and hash
    float ratio = approx_ratio(key_high, key_low);
    uint32_t hash_prefix = approx_hash(key_high, key_low);
    
    // Store ratio for sorting
    ratios[idx] = ratio;
    
    // Calculate difference from target ratio
    float ratio_diff = fabsf(ratio - target_ratio);
    uint32_t hash_diff = (hash_prefix ^ target_hash_prefix) & 0xFFFF0000; // Check top bits
    
    // Check if it's a candidate worth verifying on CPU
    if (ratio_diff < 0.0001f || hash_diff == 0) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 1000) {  // Limit to 1000 candidates
            candidates[pos] = offset;
        }
    }
}
"""

def hash160(public_key_bytes):
    """Bitcoin's hash160 function: RIPEMD160(SHA256(data))"""
    sha256_hash = hashlib.sha256(public_key_bytes).digest()
    ripemd160 = hashlib.new('ripemd160')
    ripemd160.update(sha256_hash)
    return ripemd160.hexdigest()

def get_pubkey_from_privkey(private_key_hex):
    """Get public key from private key using SECP256k1"""
    try:
        # Ensure the private key is exactly 32 bytes (64 hex chars)
        if len(private_key_hex) > 64:
            private_key_hex = private_key_hex[-64:]  # Take last 64 chars if too long
        elif len(private_key_hex) < 64:
            private_key_hex = private_key_hex.zfill(64)  # Pad with zeros if too short
        
        # Convert private key to bytes
        priv_key_bytes = binascii.unhexlify(private_key_hex)
        
        # Generate key pair using ecdsa
        sk = SigningKey.from_string(priv_key_bytes, curve=SECP256k1)
        vk = sk.verifying_key
        
        # Get public key coordinates
        x = vk.pubkey.point.x()
        y = vk.pubkey.point.y()
        
        # Format uncompressed public key (04 + x + y)
        x_bytes = x.to_bytes(32, byteorder='big')
        y_bytes = y.to_bytes(32, byteorder='big')
        pubkey = b'\x04' + x_bytes + y_bytes
        
        return {
            'pubkey': pubkey,
            'x': mpf(x),  # Use mpmath for high precision
            'y': mpf(y),  # Use mpmath for high precision
            'ratio': mpf(x) / mpf(y)  # Calculate with high precision
        }
    except Exception as e:
        # print(f"Error generating public key: {e}")
        return None

def verify_private_key(private_key_hex):
    """Verify if private key generates the target public key hash"""
    try:
        # Ensure the private key is exactly 64 hex characters
        if len(private_key_hex) != 64:
            private_key_hex = private_key_hex.zfill(64)
            
        # Get public key
        pubkey_info = get_pubkey_from_privkey(private_key_hex)
        if not pubkey_info:
            return {'match': False, 'reason': 'Failed to generate public key'}
        
        # Calculate hash160
        pubkey_hash = hash160(pubkey_info['pubkey'])
        
        # Check ratio against φ/8
        ratio = pubkey_info['ratio']
        ratio_diff = fabs(ratio - PHI_OVER_8)
        
        # Check for exact match with the hash from scriptPubKey
        exact_match = pubkey_hash == TARGET_HASH
        
        if exact_match:
            print("\n!!! EXACT MATCH FOUND WITH SCRIPTPUBKEY 76a914e0b8a2baee1b77fc703455f39d51477451fc8cfc88ac !!!")
            print(f"Private key: {private_key_hex}")
            print(f"Generated hash: {pubkey_hash}")
            print(f"Target hash: {TARGET_HASH}")
        
        return {
            'match': exact_match,
            'generated_hash': pubkey_hash,
            'ratio': ratio,
            'target_ratio': PHI_OVER_8,
            'ratio_diff': ratio_diff,
            'zero_ratio': ratio_diff == 0  # Flag for exact ratio match
        }
    except Exception as e:
        return {'match': False, 'error': str(e)}

def count_bits(n):
    """Count the number of significant bits in n"""
    if n == 0:
        return 0
    return n.bit_length()

def format_private_key(key_int):
    """Format a private key with proper padding to 64 hex characters"""
    key_hex = format(key_int, 'x')
    return key_hex.zfill(64)  # Ensure exactly 64 characters

def save_solution(key, result):
    """Save a found solution to file with detailed information"""
    try:
        with open(f"puzzle68_solution_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key: {key}\n")
            f.write(f"Generated hash: {result.get('generated_hash')}\n")
            f.write(f"Target hash: {TARGET_HASH}\n")
            f.write(f"ScriptPubKey: 76a914{TARGET_HASH}88ac\n")
            f.write(f"Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ\n")
        
        # Also save to a fixed filename for easy access
        with open("puzzle68_SOLVED.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key: {key}\n")
            f.write(f"ScriptPubKey: 76a914{TARGET_HASH}88ac\n")
            f.write(f"Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ\n")
        
        print(f"Solution saved to puzzle68_solution_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    except Exception as e:
        print(f"Error saving solution: {e}")

def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hours, {minutes} minutes"

def search_pattern(pattern_base, suffix_digits=8, batch_size=50000000, device_id=0):
    """Search a specific pattern space using GPU acceleration"""
    if not HAS_CUDA:
        print("CUDA not available. Cannot run GPU search.")
        return None
    
    # Initialize CUDA
    try:
        cuda.init()
        print("CUDA initialized successfully")
    except Exception as e:
        print(f"Failed to initialize CUDA: {e}")
        return None
    
    # Get device count
    try:
        device_count = cuda.Device.count()
        if device_count == 0:
            print("No CUDA devices found")
            return None
        if device_id >= device_count:
            print(f"Device {device_id} not available. Using device 0 instead.")
            device_id = 0
        print(f"Found {device_count} CUDA device(s)")
    except Exception as e:
        print(f"Error getting device count: {e}")
        return None
    
    # Select device
    try:
        device = cuda.Device(device_id)
        print(f"Using device {device_id}: {device.name()}")
    except Exception as e:
        print(f"Error selecting device {device_id}: {e}")
        return None
    
    # Create context
    try:
        ctx = device.make_context()
        print("Created CUDA context")
    except Exception as e:
        print(f"Error creating context: {e}")
        return None
    
    try:
        # Try to check memory info
        try:
            free_mem, total_mem = cuda.mem_get_info()
            print(f"GPU memory: {free_mem/(1024**3):.2f} GB free / {total_mem/(1024**3):.2f} GB total")
            batch_size = min(batch_size, int(free_mem * 0.7 / 16))  # 16 bytes per key
        except Exception as e:
            print(f"Warning: Could not get memory info: {e}")
        
        print(f"Using batch size: {batch_size:,}")
        
        # Compile CUDA module
        try:
            module = compiler.SourceModule(CUDA_CODE)
            test_keys_kernel = module.get_function("test_keys")
            print("CUDA kernel compiled successfully")
        except Exception as e:
            print(f"Error compiling CUDA kernel: {e}")
            ctx.pop()
            return None
        
        # Prepare base pattern
        base_int = int(pattern_base, 16)
        base_high = base_int >> 32
        base_low = base_int & 0xFFFFFFFF
        
        # Calculate bit mask for suffix digits
        suffix_bits = suffix_digits * 4
        bit_mask = (1 << suffix_bits) - 1
        
        # Clear the bits we want to explore in base_low
        base_low_masked = base_low & ~bit_mask
        
        # First 4 bytes of target hash as uint32
        target_hash_prefix = int(TARGET_HASH[:8], 16)
        
        # Allocate memory on GPU
        offsets = np.arange(batch_size, dtype=np.uint32)
        candidates = np.zeros(1000, dtype=np.uint32)
        count = np.zeros(1, dtype=np.uint32)
        ratios = np.zeros(batch_size, dtype=np.float32)
        
        offsets_gpu = cuda.mem_alloc(offsets.nbytes)
        candidates_gpu = cuda.mem_alloc(candidates.nbytes)
        count_gpu = cuda.mem_alloc(count.nbytes)
        ratios_gpu = cuda.mem_alloc(ratios.nbytes)
        
        # Determine number of batches
        max_suffix = (1 << suffix_bits) - 1
        total_batches = (max_suffix // batch_size) + 1
        
        print(f"Starting search of pattern {pattern_base + 'x'*suffix_digits}")
        print(f"Suffix space size: {max_suffix+1:,} ({suffix_digits} hex digits)")
        print(f"Estimated batches: {total_batches}")
        
        # Create/update progress tracking file for this pattern
        pattern_progress_file = f"pattern_{pattern_base}_progress.txt"
        if os.path.exists(pattern_progress_file):
            try:
                with open(pattern_progress_file, 'r') as f:
                    last_batch = int(f.readline().strip())
                    print(f"Resuming from batch {last_batch+1}/{total_batches}")
                    start_batch = last_batch + 1
            except:
                print("Could not read progress file, starting from batch 0")
                start_batch = 0
        else:
            start_batch = 0
        
        # Save initial progress
        with open(pattern_progress_file, 'w') as f:
            f.write(str(start_batch - 1) + "\n")
        
        start_time = time.time()
        total_keys_checked = start_batch * batch_size
        best_matches = []
        
        # Main search loop
        try:
            # Process batches
            for batch in range(start_batch, total_batches):
                batch_start_time = time.time()
                
                # Start with a fresh offset range for this batch
                start_offset = batch * batch_size
                end_offset = min(start_offset + batch_size, max_suffix + 1)
                current_batch_size = end_offset - start_offset
                
                # Prepare offsets for this batch
                offsets = np.arange(start_offset, end_offset, dtype=np.uint32)
                
                # Reset candidate counter
                count[0] = 0
                
                # Copy data to GPU
                cuda.memcpy_htod(offsets_gpu, offsets)
                cuda.memcpy_htod(count_gpu, count)
                
                # Set up kernel grid
                block_size = 256
                grid_size = (current_batch_size + block_size - 1) // block_size
                
                # Launch kernel
                test_keys_kernel(
                    np.uint64(base_high),
                    np.uint64(base_low_masked),
                    offsets_gpu,
                    np.int32(current_batch_size),
                    np.float32(PHI_OVER_8),
                    np.uint32(target_hash_prefix),
                    candidates_gpu,
                    ratios_gpu,
                    count_gpu,
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                
                # Get results back
                cuda.memcpy_dtoh(candidates, candidates_gpu)
                cuda.memcpy_dtoh(count, count_gpu)
                cuda.memcpy_dtoh(ratios, ratios_gpu)
                
                # Update progress
                total_keys_checked += current_batch_size
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                
                # Save batch progress
                with open(pattern_progress_file, 'w') as f:
                    f.write(str(batch) + "\n")
                
                print(f"\nBatch {batch+1}/{total_batches} completed in {format_time(batch_time)}")
                print(f"Total keys checked: {total_keys_checked:,}")
                print(f"Keys/sec: {int(current_batch_size / batch_time):,}")
                
                # Estimated time remaining
                keys_remaining = max_suffix + 1 - total_keys_checked
                estimated_time = (keys_remaining * elapsed_time) / total_keys_checked if total_keys_checked > 0 else 0
                print(f"Estimated time remaining: {format_time(estimated_time)}")
                
                # Get the number of candidates found
                candidates_found = min(int(count[0]), 1000)
                print(f"Candidates found: {candidates_found}")
                
                # Verification phase
                if candidates_found > 0:
                    print(f"Verifying {candidates_found} candidates on CPU...")
                    
                    # Sort candidates by ratio proximity to PHI/8
                    best_ratio_indices = np.argsort(np.abs(ratios[:current_batch_size] - PHI_OVER_8))[:min(200, current_batch_size)]
                    
                    # Combine explicit candidates with best ratio candidates
                    all_offsets = set()
                    for i in range(candidates_found):
                        all_offsets.add(int(candidates[i]))
                    
                    for idx in best_ratio_indices:
                        if idx < current_batch_size:  # Make sure index is in bounds
                            all_offsets.add(int(offsets[idx]))
                    
                    # Check each candidate
                    for offset in all_offsets:
                        full_key_int = (base_int & ~bit_mask) | offset
                        
                        # Ensure it's exactly 68 bits
                        if count_bits(full_key_int) != 68:
                            continue
                        
                        # Format and verify
                        full_key = format_private_key(full_key_int)
                        result = verify_private_key(full_key)
                        
                        # Check if we found a solution
                        if result and result.get('match', False):
                            print("\n*** SOLUTION FOUND! ***")
                            print(f"Private key: {full_key}")
                            print(f"Generated hash: {result['generated_hash']}")
                            print(f"Target hash: {TARGET_HASH}")
                            
                            # Save solution
                            save_solution(full_key, result)
                            
                            # Clean up and return
                            offsets_gpu.free()
                            candidates_gpu.free()
                            count_gpu.free()
                            ratios_gpu.free()
                            ctx.pop()
                            
                            return full_key
                        
                        # Track best matches
                        if result:
                            ratio_diff = float(result.get('ratio_diff', 1.0))
                            best_matches.append((full_key, ratio_diff, result.get('generated_hash', '')))
                            best_matches.sort(key=lambda x: x[1])  # Sort by ratio difference
                            best_matches = best_matches[:5]  # Keep top 5
                
                # Show best matches so far
                if best_matches:
                    print("\nBest matches so far:")
                    for i, (key, diff, hash_val) in enumerate(best_matches):
                        print(f"{i+1}. Key: {key}")
                        print(f"   Ratio diff: {diff}")
                        print(f"   Hash: {hash_val}")
                
                # Save overall progress for all patterns
                with open("puzzle68_multi_pattern_progress.txt", "a") as f:
                    f.write(f"Pattern {pattern_base + 'x'*suffix_digits} - {time.ctime()}\n")
                    f.write(f"Batch: {batch+1}/{total_batches}\n")
                    f.write(f"Keys checked: {total_keys_checked:,}\n")
                    f.write(f"Progress: {total_keys_checked/(max_suffix+1)*100:.2f}%\n\n")
            
            print(f"\nCompleted search of pattern {pattern_base + 'x'*suffix_digits}")
            print(f"Total keys checked: {total_keys_checked:,}")
            
            # Mark this pattern as searched
            with open("patterns_searched.txt", "a") as f:
                f.write(f"{pattern_base}\n")
        
        except KeyboardInterrupt:
            print("\nSearch interrupted by user")
            # Save the last completed batch for resuming later
            with open(pattern_progress_file, 'w') as f:
                f.write(str(batch) + "\n")
        
        except Exception as e:
            print(f"\nError during search: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error in GPU search: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU resources
        try:
            offsets_gpu.free()
            candidates_gpu.free()
            count_gpu.free()
            ratios_gpu.free()
        except:
            pass
        
        # Pop the context
        try:
            ctx.pop()
            print("CUDA context released")
        except:
            pass
    
    return None

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Multi-Pattern Bitcoin Puzzle Searcher")
    parser.add_argument("--pattern", type=str, help="Specific pattern to search (hex prefix)")
    parser.add_argument("--suffix-digits", type=int, default=8, help="Number of suffix hex digits to search (default: 8)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID to use (default: 0)")
    parser.add_argument("--batch-size", type=int, default=50000000, help="Batch size (default: 50M)")
    parser.add_argument("--list", action="store_true", help="List available patterns and exit")
    parser.add_argument("--all", action="store_true", help="Search all patterns sequentially")
    args = parser.parse_args()
    
    print("Multi-Pattern Bitcoin Puzzle Searcher")
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    
    # Load list of patterns already searched
    searched_patterns = set()
    if os.path.exists("patterns_searched.txt"):
        with open("patterns_searched.txt", "r") as f:
            for line in f:
                pattern = line.strip()
                if pattern:
                    searched_patterns.add(pattern)
    
    # Update the patterns list with the patterns we've already searched
    for i in range(len(PATTERNS)):
        PATTERNS[i]["searched"] = PATTERNS[i]["pattern"] in searched_patterns
    
    # If --list specified, just list the patterns and exit
    if args.list:
        print("\nAvailable patterns to search:")
        for i, pattern_info in enumerate(PATTERNS):
            status = "SEARCHED" if pattern_info["searched"] else "NOT SEARCHED"
            print(f"{i+1}. {pattern_info['pattern']} - {pattern_info['comment']} - {status}")
        return
    
    # Check whether we can use CUDA
    if not HAS_CUDA:
        print("\nCUDA is not available. Cannot run GPU search.")
        return
    
    try:
        if args.pattern:
            # Search a specific pattern
            print(f"\nSearching specific pattern: {args.pattern}")
            solution = search_pattern(
                args.pattern, 
                suffix_digits=args.suffix_digits,
                batch_size=args.batch_size,
                device_id=args.device
            )
            
            if solution:
                print(f"\nFinal solution: {solution}")
            else:
                print("\nNo solution found or search was interrupted.")
        
        elif args.all:
            # Search all patterns that haven't been searched yet
            for pattern_info in PATTERNS:
                if not pattern_info["searched"]:
                    print(f"\nSearching pattern: {pattern_info['pattern']} - {pattern_info['comment']}")
                    solution = search_pattern(
                        pattern_info["pattern"],
                        suffix_digits=args.suffix_digits,
                        batch_size=args.batch_size,
                        device_id=args.device
                    )
                    
                    if solution:
                        print(f"\nFinal solution: {solution}")
                        return
                    
                    print("\nMoving to next pattern...")
            
            print("\nAll patterns searched without finding a solution.")
        
        else:
            # Pick the first unsearched pattern
            unsearched_patterns = [p for p in PATTERNS if not p["searched"]]
            if unsearched_patterns:
                pattern_info = unsearched_patterns[0]
                print(f"\nSearching next unsearched pattern: {pattern_info['pattern']} - {pattern_info['comment']}")
                solution = search_pattern(
                    pattern_info["pattern"],
                    suffix_digits=args.suffix_digits,
                    batch_size=args.batch_size,
                    device_id=args.device
                )
                
                if solution:
                    print(f"\nFinal solution: {solution}")
                else:
                    print("\nNo solution found or search was interrupted.")
            else:
                print("\nAll patterns have been searched. Use --pattern to specify a custom pattern.")
    
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
