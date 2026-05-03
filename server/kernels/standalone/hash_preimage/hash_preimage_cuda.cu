// Standalone CUDA hash preimage search.
// Mirrors the existing WGSL hash-preimage task structure:
//   sha256(prefix || uint64_le(nonce))
// with a single-block SHA-256 message layout (prefix <= 47 bytes).
//
// Build:
//   make
// or:
//   nvcc -O3 -std=c++17 -arch=sm_70 -o hash-preimage-search hash_preimage_cuda.cu
//
// Example:
//   ./hash-preimage-search --prefix=abc --target-hash=<64-hex> --start-nonce=0 --total-nonces=512 --chunk-size=128

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err__) << std::endl; \
        std::exit(1); \
    } \
} while (0)

struct Digest {
    u32 h0, h1, h2, h3, h4, h5, h6, h7;
};

__device__ __constant__ u32 SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static const u32 SHA256_K_HOST[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

__host__ __device__ inline u32 rotr(u32 x, u32 n) {
    return (x >> n) | (x << (32u - n));
}

__host__ __device__ inline u32 ch(u32 x, u32 y, u32 z) {
    return (x & y) ^ ((~x) & z);
}

__host__ __device__ inline u32 maj(u32 x, u32 y, u32 z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__host__ __device__ inline u32 big_sigma0(u32 x) {
    return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u);
}

__host__ __device__ inline u32 big_sigma1(u32 x) {
    return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u);
}

__host__ __device__ inline u32 small_sigma0(u32 x) {
    return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u);
}

__host__ __device__ inline u32 small_sigma1(u32 x) {
    return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u);
}

template <typename ParamsPtr>
__host__ __device__ inline u32 prefix_byte_generic(const ParamsPtr params, u32 index) {
    u32 word = params[8u + (index >> 2u)];
    return (word >> ((index & 3u) * 8u)) & 0xffu;
}

__host__ __device__ inline u32 nonce_byte(u32 index, u32 nonce_lo, u32 nonce_hi) {
    if (index < 4u) return (nonce_lo >> (index * 8u)) & 0xffu;
    return (nonce_hi >> ((index - 4u) * 8u)) & 0xffu;
}

template <typename ParamsPtr>
__host__ __device__ inline u32 message_byte_generic(const ParamsPtr params, u32 index, u32 nonce_lo, u32 nonce_hi) {
    u32 prefix_len = params[27];
    u32 message_len = prefix_len + 8u;
    if (index < prefix_len) return prefix_byte_generic(params, index);
    if (index < message_len) return nonce_byte(index - prefix_len, nonce_lo, nonce_hi);
    if (index == message_len) return 0x80u;
    if (index == 62u) return ((message_len * 8u) >> 8u) & 0xffu;
    if (index == 63u) return (message_len * 8u) & 0xffu;
    return 0u;
}

template <typename ParamsPtr>
__host__ __device__ inline u32 message_word_generic(const ParamsPtr params, u32 word_index, u32 nonce_lo, u32 nonce_hi) {
    u32 base = word_index * 4u;
    return (message_byte_generic(params, base + 0u, nonce_lo, nonce_hi) << 24u) |
           (message_byte_generic(params, base + 1u, nonce_lo, nonce_hi) << 16u) |
           (message_byte_generic(params, base + 2u, nonce_lo, nonce_hi) << 8u) |
           message_byte_generic(params, base + 3u, nonce_lo, nonce_hi);
}

__host__ __device__ inline u32 digest_word(const Digest& h, u32 i) {
    switch (i) {
        case 0u: return h.h0;
        case 1u: return h.h1;
        case 2u: return h.h2;
        case 3u: return h.h3;
        case 4u: return h.h4;
        case 5u: return h.h5;
        case 6u: return h.h6;
        default: return h.h7;
    }
}

template <typename ParamsPtr>
__host__ __device__ bool leading_zero_match_generic(const ParamsPtr params, const Digest& h) {
    u32 remaining = params[28];
    for (u32 i = 0; i < 8u; i++) {
        if (remaining == 0u) return true;
        if (remaining >= 32u) {
            if (digest_word(h, i) != 0u) return false;
            remaining -= 32u;
        } else {
            u32 mask = 0xffffffffu << (32u - remaining);
            return (digest_word(h, i) & mask) == 0u;
        }
    }
    return true;
}

template <typename ParamsPtr>
__host__ __device__ bool target_match_generic(const ParamsPtr params, const Digest& h) {
    return h.h0 == params[0] &&
           h.h1 == params[1] &&
           h.h2 == params[2] &&
           h.h3 == params[3] &&
           h.h4 == params[4] &&
           h.h5 == params[5] &&
           h.h6 == params[6] &&
           h.h7 == params[7];
}

template <typename ParamsPtr>
__host__ __device__ Digest sha256_one_block_generic(const ParamsPtr params, const u32* k_table, u32 nonce_lo, u32 nonce_hi) {
    u32 w[64];
    for (u32 i = 0; i < 16u; i++) {
        w[i] = message_word_generic(params, i, nonce_lo, nonce_hi);
    }
    for (u32 i = 16u; i < 64u; i++) {
        w[i] = small_sigma1(w[i - 2u]) + w[i - 7u] + small_sigma0(w[i - 15u]) + w[i - 16u];
    }

    u32 a = 0x6a09e667u;
    u32 b = 0xbb67ae85u;
    u32 c = 0x3c6ef372u;
    u32 d = 0xa54ff53au;
    u32 e = 0x510e527fu;
    u32 f = 0x9b05688cu;
    u32 g = 0x1f83d9abu;
    u32 h = 0x5be0cd19u;

    for (u32 i = 0; i < 64u; i++) {
        u32 t1 = h + big_sigma1(e) + ch(e, f, g) + k_table[i] + w[i];
        u32 t2 = big_sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    Digest out;
    out.h0 = a + 0x6a09e667u;
    out.h1 = b + 0xbb67ae85u;
    out.h2 = c + 0x3c6ef372u;
    out.h3 = d + 0xa54ff53au;
    out.h4 = e + 0x510e527fu;
    out.h5 = f + 0x9b05688cu;
    out.h6 = g + 0x1f83d9abu;
    out.h7 = h + 0x5be0cd19u;
    return out;
}

__global__ void search_kernel(const u32* params, u32* result) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params[26] || atomicAdd(&result[0], 0u) != 0u) return;

    u32 start_lo = params[24];
    u32 nonce_lo = start_lo + idx;
    u32 carry = (nonce_lo < start_lo) ? 1u : 0u;
    u32 nonce_hi = params[25] + carry;

    Digest h = sha256_one_block_generic(params, SHA256_K, nonce_lo, nonce_hi);

    bool ok = true;
    if (params[29] != 0u) ok = ok && target_match_generic(params, h);
    if (params[28] > 0u) ok = ok && leading_zero_match_generic(params, h);
    if (!ok) return;

    if (atomicCAS(&result[0], 0u, 1u) == 0u) {
        result[1] = nonce_lo;
        result[2] = nonce_hi;
        result[3] = h.h0;
        result[4] = h.h1;
        result[5] = h.h2;
        result[6] = h.h3;
        result[7] = h.h4;
        result[8] = h.h5;
        result[9] = h.h6;
        result[10] = h.h7;
        result[11] = 0u;
    }
}

struct Config {
    std::vector<uint8_t> prefix;
    std::array<u32, 8> targetWords{};
    std::string targetHash;
    u32 leadingZeroBits = 0;
    u64 startNonce = 0;
    u64 endNonce = 0;
    u64 chunkSize = 1'000'000;
    int device = 0;
    std::string outPath;
    std::string csvPath;
    bool hasExpectedNonce = false;
    u64 expectedNonce = 0;
};

static std::unordered_map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::unordered_map<std::string, std::string> out;
    for (int i = 1; i < argc; i++) {
        std::string token(argv[i]);
        if (token.rfind("--", 0) != 0) continue;
        auto pos = token.find('=');
        if (pos == std::string::npos) out[token.substr(2)] = "1";
        else out[token.substr(2, pos - 2)] = token.substr(pos + 1);
    }
    return out;
}

static std::string getArg(const std::unordered_map<std::string, std::string>& args, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        auto it = args.find(key);
        if (it != args.end()) return it->second;
    }
    return "";
}

static u64 parseU64(const std::string& text, u64 fallback, bool* used = nullptr) {
    if (text.empty()) return fallback;
    if (used) *used = true;
    size_t idx = 0;
    u64 value = std::stoull(text, &idx, 0);
    if (idx != text.size()) throw std::runtime_error("Invalid integer: " + text);
    return value;
}

static std::vector<uint8_t> parseHexBytes(const std::string& text) {
    std::string hex = text;
    if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) hex = hex.substr(2);
    if (hex.size() % 2 != 0) throw std::runtime_error("prefixHex must contain an even number of hex characters");
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

static std::array<u32, 8> parseTargetWords(const std::string& text) {
    std::array<u32, 8> out{};
    if (text.empty()) return out;
    std::string hex = text;
    if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) hex = hex.substr(2);
    if (hex.size() != 64) throw std::runtime_error("targetHash must be 64 hex characters");
    for (size_t i = 0; i < 8; i++) {
        out[i] = static_cast<u32>(std::stoul(hex.substr(i * 8, 8), nullptr, 16));
    }
    return out;
}

static std::array<u32, 16> packPrefixWords(const std::vector<uint8_t>& prefix) {
    if (prefix.size() > 47) throw std::runtime_error("prefix must be at most 47 bytes");
    std::array<u32, 16> words{};
    for (size_t i = 0; i < prefix.size(); i++) {
        words[i >> 2] |= static_cast<u32>(prefix[i]) << ((i & 3u) * 8u);
    }
    return words;
}

static std::string digestToHex(const u32* digestWords) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 8; i++) {
        oss << std::setw(8) << (digestWords[i] & 0xffffffffu);
    }
    return oss.str();
}

static std::string nonceToHex(u64 nonce) {
    std::ostringstream oss;
    oss << "0x" << std::hex << nonce;
    return oss.str();
}

static std::string digestToHex(const Digest& d) {
    const u32 words[8] = { d.h0, d.h1, d.h2, d.h3, d.h4, d.h5, d.h6, d.h7 };
    return digestToHex(words);
}

struct TimingResults {
    std::string timestamp;
    std::string gpuName;
    size_t prefixBytes = 0;
    std::string targetHash;
    u32 leadingZeroBits = 0;
    u64 startNonce = 0;
    u64 endNonce = 0;
    u64 searchedNonces = 0;
    u64 chunkSize = 0;
    u64 batches = 0;
    bool found = false;
    double totalH2DMs = 0.0;
    double totalKernelMs = 0.0;
    double totalD2HMs = 0.0;
    double totalOutputIOMs = 0.0;
    double avgChunkTimeMs = 0.0;
    double endToEndMs = 0.0;
    double throughputNoncesPerSec = 0.0;
};

static std::string timestampHumanNow() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

static std::string generateCsvFilename(const std::string& custom = "") {
    if (!custom.empty()) return custom;
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    auto epochMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::ostringstream ss;
    ss << "hash_preimage_timing_"
       << (tm.tm_year + 1900)
       << std::setw(2) << std::setfill('0') << (tm.tm_mon + 1)
       << std::setw(2) << std::setfill('0') << tm.tm_mday
       << "_"
       << std::setw(2) << std::setfill('0') << tm.tm_hour
       << std::setw(2) << std::setfill('0') << tm.tm_min
       << std::setw(2) << std::setfill('0') << tm.tm_sec
       << "_" << (epochMs % 1000)
       << ".csv";
    return ss.str();
}

static void exportTimingToCsv(const TimingResults& r, const std::string& filename) {
    bool exists = std::ifstream(filename).good();
    std::ofstream csv(filename, std::ios::app);
    if (!csv) {
        std::cerr << "Warning: Cannot create CSV file: " << filename << std::endl;
        return;
    }

    if (!exists) {
        csv << "timestamp,gpu_name,prefix_bytes,target_hash,leading_zero_bits,start_nonce,end_nonce,"
               "searched_nonces,chunk_size,batches,found,total_h2d_ms,total_kernel_ms,total_d2h_ms,"
               "output_io_ms,avg_chunk_time_ms,end_to_end_ms,throughput_nonces_per_sec\n";
    }

    csv << std::quoted(r.timestamp) << ","
        << std::quoted(r.gpuName) << ","
        << r.prefixBytes << ","
        << std::quoted(r.targetHash) << ","
        << r.leadingZeroBits << ","
        << r.startNonce << ","
        << r.endNonce << ","
        << r.searchedNonces << ","
        << r.chunkSize << ","
        << r.batches << ","
        << (r.found ? "true" : "false") << ","
        << std::fixed << std::setprecision(3)
        << r.totalH2DMs << ","
        << r.totalKernelMs << ","
        << r.totalD2HMs << ","
        << r.totalOutputIOMs << ","
        << r.avgChunkTimeMs << ","
        << r.endToEndMs << ","
        << std::setprecision(1) << r.throughputNoncesPerSec
        << "\n";

    std::cout << "Timing CSV: " << filename << "\n";
}

static void printUsage() {
    std::cout
        << "Usage: ./hash-preimage-search [options]\n"
        << "  --prefix=TEXT or --prefix-hex=HEX\n"
        << "  --target-hash=HEX64 or --expected-nonce=N or --leading-zero-bits=N\n"
        << "  --start-nonce=N            default 0\n"
        << "  --total-nonces=N           default 512\n"
        << "  --end-nonce=N              alternative to total-nonces\n"
        << "  --chunk-size=N             per-kernel batch size, default 1000000\n"
        << "  --device=N                 CUDA device index, default 0\n"
        << "  --out=PATH                 optional JSON summary path\n"
        << "  --csv-file=PATH            optional CSV filename (auto-generated if omitted)\n";
}

static Config loadConfig(int argc, char** argv) {
    auto args = parseArgs(argc, argv);
    if (args.count("help") || args.count("h")) {
        printUsage();
        std::exit(0);
    }

    Config cfg;
    std::string prefixHex = getArg(args, {"prefix-hex", "prefixHex"});
    std::string prefixText = getArg(args, {"prefix"});
    std::string targetHash = getArg(args, {"target-hash", "targetHash"});
    std::string expectedNonceText = getArg(args, {"expected-nonce", "expectedNonce"});
    std::string leadingBitsText = getArg(args, {"leading-zero-bits", "leadingZeroBits"});
    std::string outPath = getArg(args, {"out"});
    std::string csvPath = getArg(args, {"csv-file", "csvFile"});
    bool usedEnd = false;
    bool usedExpectedNonce = false;
    u64 startNonce = parseU64(getArg(args, {"start-nonce", "startNonce"}), 0);
    u64 totalNonces = parseU64(getArg(args, {"total-nonces", "totalNonces"}), 512);
    u64 endNonce = parseU64(getArg(args, {"end-nonce", "endNonce"}), 0, &usedEnd);
    u64 chunkSize = parseU64(getArg(args, {"chunk-size", "chunkSize"}), 1'000'000);
    u64 device = parseU64(getArg(args, {"device"}), 0);
    u64 leadingBits64 = parseU64(leadingBitsText, 0);
    u64 expectedNonce = parseU64(expectedNonceText, 0, &usedExpectedNonce);

    if (!prefixHex.empty()) cfg.prefix = parseHexBytes(prefixHex);
    else cfg.prefix.assign(prefixText.begin(), prefixText.end());

    if (cfg.prefix.size() > 47) throw std::runtime_error("prefix must be at most 47 bytes");
    if (leadingBits64 > 256) throw std::runtime_error("leadingZeroBits must be <= 256");
    if (targetHash.empty() && !usedExpectedNonce && leadingBits64 == 0) {
        throw std::runtime_error("Either targetHash or expectedNonce or leadingZeroBits must be provided");
    }
    if (chunkSize == 0) throw std::runtime_error("chunkSize must be > 0");

    cfg.targetHash = targetHash;
    cfg.leadingZeroBits = static_cast<u32>(leadingBits64);
    cfg.startNonce = startNonce;
    cfg.endNonce = usedEnd ? endNonce : (startNonce + totalNonces);
    cfg.chunkSize = chunkSize;
    cfg.device = static_cast<int>(device);
    cfg.outPath = outPath;
    cfg.csvPath = csvPath;
    cfg.hasExpectedNonce = usedExpectedNonce;
    cfg.expectedNonce = expectedNonce;

    if (!cfg.targetHash.empty()) {
        cfg.targetWords = parseTargetWords(cfg.targetHash);
    }

    if (cfg.endNonce <= cfg.startNonce) throw std::runtime_error("Nonce range must satisfy endNonce > startNonce");
    return cfg;
}

static Digest verifyDigest(const std::array<u32, 32>& params, u32 nonceLo, u32 nonceHi) {
    return sha256_one_block_generic(params.data(), SHA256_K_HOST, nonceLo, nonceHi);
}

static std::string buildJsonSummary(bool found,
                                    u64 searched,
                                    u64 batches,
                                    double wallMs,
                                    double totalH2DMs,
                                    double totalKernelMs,
                                    double totalD2HMs,
                                    double totalOutputIOMs,
                                    double avgChunkTimeMs,
                                    double throughputNoncesPerSec,
                                    const std::array<u32, 12>& result) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"found\": " << (found ? "true" : "false") << ",\n";
    json << "  \"searchedNonces\": \"" << searched << "\",\n";
    json << "  \"batches\": " << batches << ",\n";
    json << "  \"wallMs\": " << std::fixed << std::setprecision(3) << wallMs << ",\n";
    json << "  \"timings\": {\n";
    json << "    \"h2dMs\": " << std::fixed << std::setprecision(3) << totalH2DMs << ",\n";
    json << "    \"kernelMs\": " << std::fixed << std::setprecision(3) << totalKernelMs << ",\n";
    json << "    \"d2hMs\": " << std::fixed << std::setprecision(3) << totalD2HMs << ",\n";
    json << "    \"outputIoMs\": " << std::fixed << std::setprecision(3) << totalOutputIOMs << ",\n";
    json << "    \"avgChunkTimeMs\": " << std::fixed << std::setprecision(3) << avgChunkTimeMs << ",\n";
    json << "    \"endToEndMs\": " << std::fixed << std::setprecision(3) << wallMs << ",\n";
    json << "    \"throughputNoncesPerSec\": " << std::fixed << std::setprecision(1) << throughputNoncesPerSec << "\n";
    json << "  }";

    if (found) {
        u64 nonce = (static_cast<u64>(result[2]) << 32) | static_cast<u64>(result[1]);
        std::array<u32, 8> digestWords = {
            result[3], result[4], result[5], result[6],
            result[7], result[8], result[9], result[10]
        };
        std::string hashHex = digestToHex(digestWords.data());
        json << ",\n  \"match\": {\n";
        json << "    \"nonce\": \"" << nonce << "\",\n";
        json << "    \"nonceHex\": \"" << nonceToHex(nonce) << "\",\n";
        json << "    \"hash\": \"" << hashHex << "\"\n";
        json << "  }";
    }
    json << "\n}\n";
    return json.str();
}

int main(int argc, char** argv) {
    try {
        Config cfg = loadConfig(argc, argv);
        CUDA_CHECK(cudaSetDevice(cfg.device));

        auto prefixWords = packPrefixWords(cfg.prefix);
        if (cfg.targetHash.empty() && cfg.hasExpectedNonce) {
            std::array<u32, 32> expectedParams{};
            for (size_t i = 0; i < 16; i++) expectedParams[8 + i] = prefixWords[i];
            expectedParams[27] = static_cast<u32>(cfg.prefix.size());
            Digest expectedDigest = verifyDigest(
                expectedParams,
                static_cast<u32>(cfg.expectedNonce & 0xffffffffull),
                static_cast<u32>((cfg.expectedNonce >> 32) & 0xffffffffull)
            );
            cfg.targetHash = digestToHex(expectedDigest);
            cfg.targetWords = {
                expectedDigest.h0, expectedDigest.h1, expectedDigest.h2, expectedDigest.h3,
                expectedDigest.h4, expectedDigest.h5, expectedDigest.h6, expectedDigest.h7
            };
        }

        std::array<u32, 32> params{};
        for (size_t i = 0; i < 8; i++) params[i] = cfg.targetWords[i];
        for (size_t i = 0; i < 16; i++) params[8 + i] = prefixWords[i];
        params[27] = static_cast<u32>(cfg.prefix.size());
        params[28] = cfg.leadingZeroBits;
        params[29] = cfg.targetHash.empty() ? 0u : 1u;

        u32* d_params = nullptr;
        u32* d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_params, sizeof(u32) * 32));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(u32) * 12));

        auto t0 = std::chrono::high_resolution_clock::now();
        double totalH2DMs = 0.0;
        double totalKernelMs = 0.0;
        double totalD2HMs = 0.0;
        double totalOutputIOMs = 0.0;
        u64 searched = 0;
        u64 batches = 0;
        std::array<u32, 12> result{};
        bool found = false;
        std::vector<double> chunkTimesMs;

        for (u64 nonce = cfg.startNonce; nonce < cfg.endNonce; nonce += cfg.chunkSize) {
            auto chunkT0 = std::chrono::high_resolution_clock::now();
            u64 remaining = cfg.endNonce - nonce;
            u64 batchCount64 = std::min<u64>(cfg.chunkSize, remaining);
            batchCount64 = std::min<u64>(batchCount64, 0xffffffffull);
            u32 batchCount = static_cast<u32>(batchCount64);

            params[24] = static_cast<u32>(nonce & 0xffffffffull);
            params[25] = static_cast<u32>((nonce >> 32) & 0xffffffffull);
            params[26] = batchCount;

            cudaEvent_t evH2DStart, evH2DStop, evKernelStart, evKernelStop, evD2HStart, evD2HStop;
            CUDA_CHECK(cudaEventCreate(&evH2DStart));
            CUDA_CHECK(cudaEventCreate(&evH2DStop));
            CUDA_CHECK(cudaEventCreate(&evKernelStart));
            CUDA_CHECK(cudaEventCreate(&evKernelStop));
            CUDA_CHECK(cudaEventCreate(&evD2HStart));
            CUDA_CHECK(cudaEventCreate(&evD2HStop));

            CUDA_CHECK(cudaEventRecord(evH2DStart));
            CUDA_CHECK(cudaMemcpy(d_params, params.data(), sizeof(u32) * 32, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(u32) * 12));
            CUDA_CHECK(cudaEventRecord(evH2DStop));
            CUDA_CHECK(cudaEventSynchronize(evH2DStop));

            constexpr u32 BLOCK_SIZE = 256u;
            u32 grid = (batchCount + BLOCK_SIZE - 1u) / BLOCK_SIZE;

            CUDA_CHECK(cudaEventRecord(evKernelStart));
            if (batchCount > 0) {
                search_kernel<<<grid, BLOCK_SIZE>>>(d_params, d_result);
            }
            CUDA_CHECK(cudaEventRecord(evKernelStop));
            CUDA_CHECK(cudaEventSynchronize(evKernelStop));

            float h2dMs = 0.0f;
            float kernelMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&h2dMs, evH2DStart, evH2DStop));
            CUDA_CHECK(cudaEventElapsedTime(&kernelMs, evKernelStart, evKernelStop));
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(evD2HStart));
            CUDA_CHECK(cudaMemcpy(result.data(), d_result, sizeof(u32) * 12, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(evD2HStop));
            CUDA_CHECK(cudaEventSynchronize(evD2HStop));

            float d2hMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&d2hMs, evD2HStart, evD2HStop));

            CUDA_CHECK(cudaEventDestroy(evH2DStart));
            CUDA_CHECK(cudaEventDestroy(evH2DStop));
            CUDA_CHECK(cudaEventDestroy(evKernelStart));
            CUDA_CHECK(cudaEventDestroy(evKernelStop));
            CUDA_CHECK(cudaEventDestroy(evD2HStart));
            CUDA_CHECK(cudaEventDestroy(evD2HStop));

            totalH2DMs += h2dMs;
            totalKernelMs += kernelMs;
            totalD2HMs += d2hMs;
            searched += batchCount64;
            batches++;
            auto chunkT1 = std::chrono::high_resolution_clock::now();
            chunkTimesMs.push_back(std::chrono::duration<double, std::milli>(chunkT1 - chunkT0).count());

            if (result[0] == 1u) {
                found = true;
                break;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double wallMs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        double avgChunkTimeMs = chunkTimesMs.empty() ? 0.0 :
            (std::accumulate(chunkTimesMs.begin(), chunkTimesMs.end(), 0.0) / static_cast<double>(chunkTimesMs.size()));
        double throughputNoncesPerSec = wallMs > 0.0 ? (static_cast<double>(searched) * 1000.0 / wallMs) : 0.0;

        if (found) {
            u64 nonce = (static_cast<u64>(result[2]) << 32) | static_cast<u64>(result[1]);
            Digest verified = verifyDigest(params, result[1], result[2]);
            std::array<u32, 8> digestWords = {
                result[3], result[4], result[5], result[6],
                result[7], result[8], result[9], result[10]
            };
            std::array<u32, 8> verifiedWords = {
                verified.h0, verified.h1, verified.h2, verified.h3,
                verified.h4, verified.h5, verified.h6, verified.h7
            };
            if (digestWords != verifiedWords) {
                throw std::runtime_error("GPU result digest failed host verification");
            }

            std::string hashHex = digestToHex(digestWords.data());
            std::cout << "Found nonce: " << nonce << " (" << nonceToHex(nonce) << ")\n";
            std::cout << "Hash: " << hashHex << "\n";
        }

        CUDA_CHECK(cudaFree(d_params));
        CUDA_CHECK(cudaFree(d_result));
        std::cout << "Searched nonces: " << searched << "\n";
        std::cout << "Batches: " << batches << "\n";
        std::cout << "H2D time (ms): " << totalH2DMs << "\n";
        std::cout << "Kernel time (ms): " << totalKernelMs << "\n";
        std::cout << "D2H time (ms): " << totalD2HMs << "\n";
        std::cout << "Avg chunk time (ms): " << avgChunkTimeMs << "\n";
        std::cout << "Throughput (nonces/s): " << throughputNoncesPerSec << "\n";
        std::cout << "Wall time (ms): " << wallMs << "\n";

        if (!cfg.outPath.empty()) {
            auto outputIoT0 = std::chrono::high_resolution_clock::now();
            std::string jsonText = buildJsonSummary(found, searched, batches, wallMs, totalH2DMs, totalKernelMs,
                                                    totalD2HMs, 0.0, avgChunkTimeMs, throughputNoncesPerSec, result);
            std::ofstream out(cfg.outPath, std::ios::binary);
            if (!out) throw std::runtime_error("Failed to open output path: " + cfg.outPath);
            out << jsonText;
            auto outputIoT1 = std::chrono::high_resolution_clock::now();
            totalOutputIOMs += std::chrono::duration<double, std::milli>(outputIoT1 - outputIoT0).count();
        }

        cudaDeviceProp prop{};
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        TimingResults timing{};
        timing.timestamp = timestampHumanNow();
        timing.gpuName = prop.name;
        timing.prefixBytes = cfg.prefix.size();
        timing.targetHash = cfg.targetHash;
        timing.leadingZeroBits = cfg.leadingZeroBits;
        timing.startNonce = cfg.startNonce;
        timing.endNonce = cfg.endNonce;
        timing.searchedNonces = searched;
        timing.chunkSize = cfg.chunkSize;
        timing.batches = batches;
        timing.found = found;
        timing.totalH2DMs = totalH2DMs;
        timing.totalKernelMs = totalKernelMs;
        timing.totalD2HMs = totalD2HMs;
        timing.totalOutputIOMs = totalOutputIOMs;
        timing.avgChunkTimeMs = avgChunkTimeMs;
        timing.endToEndMs = wallMs;
        timing.throughputNoncesPerSec = throughputNoncesPerSec;
        exportTimingToCsv(timing, generateCsvFilename(cfg.csvPath));

        std::string jsonText = buildJsonSummary(found, searched, batches, wallMs, totalH2DMs, totalKernelMs,
                                                totalD2HMs, totalOutputIOMs, avgChunkTimeMs, throughputNoncesPerSec, result);
        std::cout << jsonText;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
