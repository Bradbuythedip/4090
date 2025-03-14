# puzzle68_robust_solver.py

import hashlib
import binascii
import time
import os
import random
import numpy as np
from mpmath import mp, mpf, fabs
import ecdsa
from ecdsa import SECP256k1, SigningKey
import math
from itertools import combinations
import signal
import sys
import threading
import queue
from datetime import datetime, timedelta

# Set high precision for mpmath
mp.dps = 1000  # 1000 decimal places of precision

# Constants
PHI = mpf('1.6180339887498948482045868343656381177203091798057628621354486227052604628189')
TARGET_HASH = 'e0b8a2baee1b77fc703455f39d51477451fc8cfc'  # Hash from scriptPubKey: 76a914e0b8a2baee1b77fc703455f39d51477451fc8cfc88ac
PUZZLE_NUMBER = 68
PHI_OVER_8 = float(PHI / 8)  # Convert to float for GPU compatibility

# Our best candidates from different approaches
BEST_PHI_MATCH = "00000000000000000000000000000000000000000000000041a01b90c5b886dc"
BEST_ALT_PATTERN = "00000000000000000000000000000000000000000000000cedb187f001bdffd2"

# CEDB187F pattern space (base pattern for focused search)
CEDB187F_PATTERN = "00000000000000000000000000000000000000000000000cedb187f"

# Additional promising candidates
ADDITIONAL_CANDIDATES = [
    "00000000000000000000000000000000000000000000000041a01b90c5b886dd",  # Small variation of PHI_MATCH
    "00000000000000000000000000000000000000000000000cedb187f001bdffe",   # Small variation of ALT_PATTERN
    "00000000000000000000000000000000000000000000000467a98b1c3d2e5f0",   # Additional candidate
    "00000000000000000000000000000000000000000000000789abcdef0123456",   # Additional candidate
]

# Global variables for runtime control
RUNNING = True
START_TIME = time.time()
KEYS_CHECKED = 0
BEST_GLOBAL_MATCHES = []
BEST_GLOBAL_RATIO_DIFF = 1.0
SOLUTION_FOUND = False

# Create a shared lock for thread-safe operations
GLOBAL_LOCK = threading.Lock()

# Initialize the last save time
save_progress_last_save_time = 0

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
            
            # Signal all processes to stop
            global SOLUTION_FOUND
            SOLUTION_FOUND = True
        
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
            f.write(f"Ratio: {result.get('ratio')}\n")
            f.write(f"Ratio diff from PHI/8: {result.get('ratio_diff')}\n")
            f.write(f"Total keys checked: {KEYS_CHECKED:,}\n")
            f.write(f"Search duration: {format_time(time.time() - START_TIME)}\n")
        
        # Also save to a fixed filename for easy access
        with open("puzzle68_SOLVED.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key: {key}\n")
            f.write(f"ScriptPubKey: 76a914{TARGET_HASH}88ac\n")
            f.write(f"Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ\n")
    except Exception as e:
        print(f"Error saving solution: {e}")

def save_progress(best_matches, operation="", attempt_count=0, force_save=False):
    """Save current progress and best matches to a file with more detailed stats"""
    global KEYS_CHECKED, BEST_GLOBAL_MATCHES, save_progress_last_save_time
    
    # Update total keys checked
    with GLOBAL_LOCK:
        KEYS_CHECKED += attempt_count
        
        # Update best global matches
        for key, result in best_matches:
            # Check if this is a better match than what we have
            if not BEST_GLOBAL_MATCHES or float(result.get('ratio_diff', 1.0)) < float(BEST_GLOBAL_MATCHES[0][1].get('ratio_diff', 1.0)):
                BEST_GLOBAL_MATCHES.append((key, result))
                BEST_GLOBAL_MATCHES.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                BEST_GLOBAL_MATCHES = BEST_GLOBAL_MATCHES[:10]  # Keep top 10
    
    # Don't save too frequently unless forced
    current_time = time.time()
    if not force_save and current_time - save_progress_last_save_time < 30:  # Save every 30 seconds
        return
    
    save_progress_last_save_time = current_time
    
    try:
        with open("puzzle68_progress.txt", "w") as f:
            f.write(f"Search progress as of {time.ctime()}\n")
            f.write(f"Runtime: {format_time(current_time - START_TIME)}\n")
            f.write(f"Current operation: {operation}\n")
            f.write(f"Total keys checked: {KEYS_CHECKED:,}\n")
            
            # Calculate rate
            elapsed = current_time - START_TIME
            if elapsed > 0:
                rate = int(KEYS_CHECKED / elapsed)
                f.write(f"Keys per second: {rate:,}\n")
                # Estimate time to completion for 16^10 search space
                if rate > 0:
                    total_space = 16**10  # Search space size for 10 hex digits
                    remaining = max(0, total_space - KEYS_CHECKED)
                    seconds_remaining = remaining / rate
                    f.write(f"Estimated time remaining: {format_time(seconds_remaining)}\n")
            
            f.write("\nTop closest matches by ratio:\n")
            for i, (key, result) in enumerate(BEST_GLOBAL_MATCHES):
                f.write(f"{i+1}. Key: {key}\n")
                bit_length = count_bits(int(key, 16))
                f.write(f"   Bit length: {bit_length}\n")
                f.write(f"   Ratio diff: {result.get('ratio_diff')}\n")
                f.write(f"   Generated hash: {result.get('generated_hash')}\n")
                f.write(f"   Target hash: {TARGET_HASH}\n\n")
    except Exception as e:
        print(f"Error saving progress: {e}")

def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hours, {minutes} minutes"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days} days, {hours} hours"

def try_multi_bit_manipulations(key, max_flips=5, max_positions=32):
    """Try flipping multiple bits at once in the key - enhanced for exhaustive search"""
    key_int = int(key, 16)
    print(f"Trying multi-bit manipulations of {key}")
    print(f"Max bits to flip: {max_flips}, Max positions: {max_positions}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Positions to try bit flips (focus on both lower and higher bits)
    positions = list(range(max_positions))
    
    # Try flipping 1 to max_flips bits at a time
    for num_flips in range(1, max_flips + 1):
        print(f"Trying {num_flips}-bit flips...")
        # Generate combinations of num_flips positions
        for combo in combinations(positions, num_flips):
            if not RUNNING or SOLUTION_FOUND:
                print("Search interrupted")
                return None
                
            # Start with original key
            modified_key = key_int
            
            # Flip bits at all positions in the combo
            for pos in combo:
                modified_key ^= (1 << pos)
            
            # Ensure it's still 68 bits
            if count_bits(modified_key) != 68:
                continue
            
            # Format and test
            modified_key_hex = format_private_key(modified_key)
            result = verify_private_key(modified_key_hex)
            count += 1
            
            if count % 10000 == 0:
                print(f"Tested {count:,} multi-bit variations...")
                # Save progress periodically
                save_progress(best_matches, f"Multi-bit manipulation ({num_flips} bits)", count)
            
            # Check if we found the solution
            if result and result.get('match'):
                print(f"\n*** SOLUTION FOUND ***")
                print(f"Key: {modified_key_hex}")
                print(f"Generated hash: {result.get('generated_hash')}")
                print(f"ScriptPubKey: 76a914{result.get('generated_hash')}88ac")
                
                # Save solution
                save_solution(modified_key_hex, result)
                
                # Exit immediately on match
                print("\nExiting program immediately with solution found.")
                os._exit(0)  # Force immediate exit
                
                return modified_key_hex
            
            # Track best match by ratio
            if result:
                ratio_diff = float(result.get('ratio_diff', 1.0))
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    
                    # Add to best matches
                    best_matches.append((modified_key_hex, result))
                    best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                    best_matches = best_matches[:5]  # Keep top 5
                    
                    # Report improvement
                    print(f"\nNew best match: {modified_key_hex}")
                    print(f"Bit length: {count_bits(modified_key)}")
                    print(f"Ratio diff: {ratio_diff}")
                    print(f"Generated hash: {result.get('generated_hash')}")
                    
                    # Save progress on significant improvement
                    save_progress(best_matches, f"Multi-bit manipulation ({num_flips} bits)", count, force_save=True)
    
    print(f"Completed multi-bit manipulations, tested {count:,} variations")
    return None

def precise_search_around_key(key, range_size=100000, step_size=1, max_keys=10000000):
    """Perform a precise search around a key with very small steps - enhanced for exhaustive search"""
    key_int = int(key, 16)
    print(f"Starting precise search around {key}")
    print(f"Range: +/- {range_size}, Step size: {step_size}, Max keys: {max_keys:,}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Search within range of the key
    for offset in range(-range_size, range_size + 1, step_size):
        if offset == 0:  # Skip the original key
            continue
            
        if not RUNNING or SOLUTION_FOUND:
            print("Search interrupted")
            break
            
        # Calculate modified key
        modified_key = key_int + offset
        
        # Ensure it's still 68 bits
        if count_bits(modified_key) != 68:
            continue
        
        # Format and test
        modified_key_hex = format_private_key(modified_key)
        result = verify_private_key(modified_key_hex)
        count += 1
        
        if count % 10000 == 0:
            print(f"Tested {count:,} keys in precise search...")
            # Save progress periodically
            save_progress(best_matches, "Precise search", count)
        
        # Check if we found the solution
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND ***")
            print(f"Key: {modified_key_hex}")
            print(f"Generated hash: {result.get('generated_hash')}")
            
            # Save solution
            save_solution(modified_key_hex, result)
            
            return modified_key_hex
        
        # Track best match by ratio
        if result:
            ratio_diff = float(result.get('ratio_diff', 1.0))
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                
                # Add to best matches
                best_matches.append((modified_key_hex, result))
                best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                best_matches = best_matches[:5]  # Keep top 5
                
                # Report improvement
                print(f"\nNew best match: {modified_key_hex}")
                print(f"Bit length: {count_bits(modified_key)}")
                print(f"Ratio diff: {ratio_diff}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save progress on significant improvement
                save_progress(best_matches, "Precise search", count, force_save=True)
        
        # Check if we've tested enough keys
        if count >= max_keys:
            print(f"Reached max keys limit of {max_keys:,}")
            break
    
    print(f"Completed precise search, tested {count:,} keys")
    return None

def random_bit_permutation_search(key, iterations=1000000):
    """Try random bit permutations of a key"""
    key_int = int(key, 16)
    print(f"Starting random bit permutation search from {key}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Define bit operations to try
    operations = [
        lambda k: k ^ (1 << random.randint(0, 40)),  # Flip random bit
        lambda k: k ^ (random.randint(1, 255)),     # XOR with small random value
        lambda k: (k & ~0xFF) | random.randint(0, 255),  # Randomize lowest byte
        lambda k: k + random.randint(-1000, 1000),  # Small addition/subtraction
        lambda k: ((k << 1) & ((1 << 68) - 1)) | (k >> 67),  # Rotate bits
        lambda k: k ^ (PUZZLE_NUMBER << random.randint(0, 16))  # XOR with puzzle number
    ]
    
    for i in range(iterations):
        if not RUNNING or SOLUTION_FOUND:
            print("Search interrupted")
            break
            
        # Choose random operation
        op = random.choice(operations)
        modified_key = op(key_int)
        
        # Ensure it's still 68 bits
        if count_bits(modified_key) != 68:
            continue
            
        # Format and test
        modified_key_hex = format_private_key(modified_key)
        result = verify_private_key(modified_key_hex)
        count += 1
        
        if count % 10000 == 0:
            print(f"Tested {count:,} random bit permutations...")
            save_progress(best_matches, "Random bit permutations", count)
        
        # Check if we found the solution
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND ***")
            print(f"Key: {modified_key_hex}")
            print(f"Generated hash: {result.get('generated_hash')}")
            
            # Save solution
            save_solution(modified_key_hex, result)
            
            return modified_key_hex
        
        # Track best match by ratio
        if result:
            ratio_diff = float(result.get('ratio_diff', 1.0))
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                
                # Add to best matches
                best_matches.append((modified_key_hex, result))
                best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                best_matches = best_matches[:5]  # Keep top 5
                
                # Report improvement
                print(f"\nNew best match: {modified_key_hex}")
                print(f"Bit length: {count_bits(modified_key)}")
                print(f"Ratio diff: {ratio_diff}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save progress
                save_progress(best_matches, "Random bit permutations", count, force_save=True)
    
    print(f"Completed random bit permutations, tested {count:,} variations")
    return None

def try_mathematical_transformations(key):
    """Try various mathematical transformations of the key"""
    key_int = int(key, 16)
    print(f"Trying mathematical transformations of {key}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Multipliers based on various mathematical constants
    multipliers = [
        int(PHI * 2**32) & ((1 << 68) - 1),  # PHI scaled to 68 bits
        int(math.e * 2**32) & ((1 << 68) - 1),  # e scaled to 68 bits
        int(math.pi * 2**32) & ((1 << 68) - 1),  # pi scaled to 68 bits
        int(math.sqrt(2) * 2**32) & ((1 << 68) - 1),  # sqrt(2) scaled to 68 bits
        PUZZLE_NUMBER,  # Puzzle number
        PUZZLE_NUMBER**2,  # Puzzle number squared
        int(PUZZLE_NUMBER * PHI),  # Puzzle number * PHI
    ]
    
    transformations = [
        # Addition with constants
        lambda k, m: (k + m) & ((1 << 68) - 1),
        # Subtraction with constants
        lambda k, m: (k - m) & ((1 << 68) - 1),
        # XOR with constants
        lambda k, m: k ^ m,
        # Multiply low bits with constant
        lambda k, m: (k & ~0xFFFF) | ((k & 0xFFFF) * (m & 0xFF)) & 0xFFFF,
        # Bit rotations
        lambda k, m: ((k << (m % 68)) | (k >> (68 - (m % 68)))) & ((1 << 68) - 1),
    ]
    
    for transform in transformations:
        for multiplier in multipliers:
            if not RUNNING or SOLUTION_FOUND:
                print("Search interrupted")
                return None
                
            modified_key = transform(key_int, multiplier)
            
            # Ensure it's still 68 bits
            if count_bits(modified_key) != 68:
                continue
                
            # Format and test
            modified_key_hex = format_private_key(modified_key)
            result = verify_private_key(modified_key_hex)
            count += 1
            
            # Check if we found the solution
            if result and result.get('match'):
                print(f"\n*** SOLUTION FOUND ***")
                print(f"Key: {modified_key_hex}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save solution
                save_solution(modified_key_hex, result)
                
                return modified_key_hex
            
            # Track best match by ratio
            if result:
                ratio_diff = float(result.get('ratio_diff', 1.0))
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    
                    # Add to best matches
                    best_matches.append((modified_key_hex, result))
                    best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                    best_matches = best_matches[:5]  # Keep top 5
                    
                    # Report improvement
                    print(f"\nNew best match: {modified_key_hex}")
                    print(f"Bit length: {count_bits(modified_key)}")
                    print(f"Ratio diff: {ratio_diff}")
                    print(f"Generated hash: {result.get('generated_hash')}")
                    
                    # Save progress
                    save_progress(best_matches, "Mathematical transformations", count, force_save=True)
    
    print(f"Completed mathematical transformations, tested {count:,} variations")
    save_progress(best_matches, "Mathematical transformations", count)
    return None

def handle_exit_signal(sig, frame):
    """Handle user exit signal (Ctrl+C)"""
    global RUNNING
    if not RUNNING:
        print("\nForce exiting...")
        sys.exit(1)
    
    print("\nGracefully shutting down... (Press Ctrl+C again to force exit)")
    RUNNING = False

def systematic_pattern_search(pattern_base, target_digits=10, step=1, max_tests=1000000):
    """Systematically search through a pattern space"""
    print(f"Starting systematic search of pattern: {pattern_base}xxxxxxxxxx")
    base_int = int(pattern_base, 16)
    
    # Calculate search space boundaries
    suffix_bits = target_digits * 4
    suffix_max = (1 << suffix_bits) - 1
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Search through the suffix space in steps
    for suffix in range(0, suffix_max + 1, step):
        if not RUNNING or SOLUTION_FOUND:
            print("Search interrupted")
            break
            
        # Create full key
        full_key_int = base_int | suffix
        
        # Ensure it's 68 bits
        if count_bits(full_key_int) != 68:
            continue
        
        # Format and test
        full_key_hex = format_private_key(full_key_int)
        result = verify_private_key(full_key_hex)
        count += 1
        
        if count % 10000 == 0:
            print(f"Tested {count:,} patterns...")
            suffix_hex = format(suffix, f'0{target_digits}x')
            current = f"{pattern_base}{suffix_hex}"
            print(f"Currently at: {current}")
            save_progress(best_matches, "Systematic pattern search", count)
        
        # Check if we found the solution
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND ***")
            print(f"Key: {full_key_hex}")
            print(f"Generated hash: {result.get('generated_hash')}")
            
            # Save solution
            save_solution(full_key_hex, result)
            
            return full_key_hex
        
        # Track best match by ratio
        if result:
            ratio_diff = float(result.get('ratio_diff', 1.0))
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                
                # Add to best matches
                best_matches.append((full_key_hex, result))
                best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                best_matches = best_matches[:5]  # Keep top 5
                
                # Report improvement
                print(f"\nNew best match: {full_key_hex}")
                print(f"Bit length: {count_bits(full_key_int)}")
                print(f"Ratio diff: {ratio_diff}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save progress
                save_progress(best_matches, "Systematic pattern search", count, force_save=True)
        
        # Check if we've tested enough keys
        if count >= max_tests:
            print(f"Reached max tests limit of {max_tests:,}")
            break
    
    print(f"Completed systematic pattern search, tested {count:,} keys")
    return None

def search_cedb187f_pattern():
    """Focus search on the CEDB187F pattern space"""
    print(f"Starting search of CEDB187F pattern space")
    pattern = CEDB187F_PATTERN
    
    # Try a systematic search with step size to cover more ground
    step_size = 1001  # Use a prime number to ensure good coverage
    return systematic_pattern_search(pattern, target_digits=10, step=step_size, max_tests=10000000)

def partition_search_space(num_threads=4):
    """Partition the search space for multi-threaded search"""
    total_space = 16**10  # 10 hex digits
    keys_per_thread = total_space // num_threads
    
    partitions = []
    for i in range(num_threads):
        start = i * keys_per_thread
        end = (i+1) * keys_per_thread if i < num_threads-1 else total_space
        
        start_hex = format(start, '010x')
        end_hex = format(end, '010x')
        
        partitions.append({
            'start': start,
            'end': end,
            'start_hex': start_hex,
            'end_hex': end_hex,
            'pattern': f"{CEDB187F_PATTERN}{start_hex}"
        })
    
    return partitions

def threaded_pattern_search(thread_id, start, end, pattern):
    """Search a specific range of the pattern space in a thread"""
    print(f"Thread {thread_id}: Searching from {start:x} to {end:x}")
    
    base_int = int(CEDB187F_PATTERN, 16)
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Search through the range systematically with a step
    step = 997  # A prime number step helps ensure good coverage
    
    for suffix in range(start, end, step):
        if not RUNNING or SOLUTION_FOUND:
            print(f"Thread {thread_id}: Search interrupted")
            break
            
        # Create full key
        full_key_int = base_int | suffix
        
        # Ensure it's 68 bits
        if count_bits(full_key_int) != 68:
            continue
        
        # Format and test
        full_key_hex = format_private_key(full_key_int)
        result = verify_private_key(full_key_hex)
        count += 1
        
        if count % 10000 == 0:
            percent = (suffix - start) / (end - start) * 100 if end > start else 0
            print(f"Thread {thread_id}: {percent:.1f}% complete, tested {count:,} keys")
            save_progress(best_matches, f"Thread {thread_id} search", count)
        
        # Check if we found the solution
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND in Thread {thread_id} ***")
            print(f"Key: {full_key_hex}")
            print(f"Generated hash: {result.get('generated_hash')}")
            
            # Save solution
            save_solution(full_key_hex, result)
            global SOLUTION_FOUND
            SOLUTION_FOUND = True
            
            return full_key_hex
        
        # Track best match by ratio
        if result:
            ratio_diff = float(result.get('ratio_diff', 1.0))
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                
                # Add to best matches
                best_matches.append((full_key_hex, result))
                best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                best_matches = best_matches[:5]  # Keep top 5
                
                # Report improvement
                print(f"\nThread {thread_id}: New best match: {full_key_hex}")
                print(f"Bit length: {count_bits(full_key_int)}")
                print(f"Ratio diff: {ratio_diff}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save progress
                save_progress(best_matches, f"Thread {thread_id} search", count, force_save=True)
    
    print(f"Thread {thread_id}: Completed, tested {count:,} keys")
    return None

def multi_threaded_search(num_threads=8):
    """Run a multi-threaded search of the pattern space"""
    print(f"Starting multi-threaded search with {num_threads} threads")
    
    # Partition the search space
    partitions = partition_search_space(num_threads)
    
    # Create and start threads
    threads = []
    for i, part in enumerate(partitions):
        thread = threading.Thread(
            target=threaded_pattern_search,
            args=(i, part['start'], part['end'], part['pattern'])
        )
        thread.start()
        threads.append(thread)
        
        # Small delay between thread starts
        time.sleep(0.5)
    
    # Wait for threads to complete
    try:
        while True:
            if all(not t.is_alive() for t in threads) or SOLUTION_FOUND:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
        global RUNNING
        RUNNING = False
        
        # Wait for threads to notice and exit
        for t in threads:
            t.join()
    
    print("All search threads completed")
    return None

def try_known_candidates():
    """Try all known candidate keys first"""
    print("\nVerifying known candidate keys...")
    for key in [BEST_PHI_MATCH, BEST_ALT_PATTERN] + ADDITIONAL_CANDIDATES:
        print(f"Checking {key}...")
        result = verify_private_key(key)
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND IN INITIAL VERIFICATION ***")
            save_solution(key, result)
            return key
    return None

def main_search():
    """Main search function that runs multiple strategies"""
    global RUNNING
    signal.signal(signal.SIGINT, handle_exit_signal)
    
    print("\nStarting main search process...")
    
    # First try all known candidates
    solution = try_known_candidates()
    if solution:
        return solution
    
    # Try systematic search of CEDB187F pattern with multiple threads
    print("\nStarting multi-threaded search of CEDB187F pattern space...")
    solution = multi_threaded_search(num_threads=8)  # Use 8 threads for CPU search
    if solution:
        return solution
    
    # If pattern search didn't find a solution, try other strategies
    search_strategies = [
        # Try precise search around best matches
        lambda: precise_search_around_key(BEST_PHI_MATCH, range_size=500000, step_size=1),
        lambda: precise_search_around_key(BEST_ALT_PATTERN, range_size=500000, step_size=1),
        
        # Try bit manipulations
        lambda: try_multi_bit_manipulations(BEST_PHI_MATCH, max_flips=4, max_positions=32),
        lambda: try_multi_bit_manipulations(BEST_ALT_PATTERN, max_flips=4, max_positions=32),
        
        # Try random search
        lambda: random_bit_permutation_search(BEST_PHI_MATCH, iterations=2000000),
        lambda: random_bit_permutation_search(BEST_ALT_PATTERN, iterations=2000000),
        
        # Try math transformations
        lambda: try_mathematical_transformations(BEST_PHI_MATCH),
        lambda: try_mathematical_transformations(BEST_ALT_PATTERN),
    ]
    
    # Run each strategy until solution is found or all are exhausted
    for i, strategy in enumerate(search_strategies):
        if not RUNNING:
            break
            
        print(f"\n--- Strategy {i+1}/{len(search_strategies)} ---")
        solution = strategy()
        if solution:
            return solution
    
    return None

if __name__ == "__main__":
    print("Bitcoin Puzzle #68 Solver - Robust Edition")
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    print(f"Primary search pattern: {CEDB187F_PATTERN}")
    
    # Initialize start time
    START_TIME = time.time()
    
    print("\nRunning CPU-based search (no GPU required)...")
    print("Press Ctrl+C to gracefully stop the search")
    
    try:
        # Run the main search
        solution = main_search()
        
        elapsed_time = time.time() - START_TIME
        print(f"\nSearch completed in {format_time(elapsed_time)}")
        
        if solution:
            print(f"\nFinal solution: {solution}")
            sys.exit(0)
        else:
            print("\nNo solution found in the search space.")
            print("Check puzzle68_progress.txt for the best matches so far.")
    
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
        elapsed_time = time.time() - START_TIME
        print(f"Search ran for {format_time(elapsed_time)}")
        print(f"Total keys checked: {KEYS_CHECKED:,}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError during search: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
