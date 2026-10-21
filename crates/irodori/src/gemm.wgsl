// C[M,N] = A[M,K] x B[K,N](行優先・連続、f32)。K と N は 4 の倍数(ホスト側で確認する)。
// ワークグループ 16x16。各スレッドが TM 行 x TN 列を受け持つ(行は 16 刻み、列は vec4 を 16 刻み)。
// タイル BM x BK(A)と BK x BN(B)を共有メモリに置き、BK ずつ K を進める。
// 累積はレジスタに置くため、行・列のループは Rust 側(gemm.rs)で展開して埋め込む。
// K 分割(split-K): ワークグループの z が K の区間 [z*kchunk, (z+1)*kchunk) を受け持ち、部分和を c の z 番目の
// [M,N] に書く(分割しないときは z = 0 の 1 つだけ)。部分和の合計は reduce.wgsl。
// バッチ(注意のヘッドごとの行列積など): z = バッチ * splits + 分割。A・B はバッチごとに a_bs・b_bs(vec4 単位)ずつずらし、
// 出力は [splits, バッチ, M, N]。

@group(0) @binding(0) var<storage, read_write> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dims: array<u32>;

const BM: u32 = {{ bm }}u;
const BNV: u32 = {{ bnv }}u;
const BK: u32 = {{ bk }}u;

// As は転置して [BK][BM](スカラー)、Bs は [BK][BN/4](vec4)
var<workgroup> As: array<f32, {{ as_len }}>;
var<workgroup> Bs: array<vec4<f32>, {{ bs_len }}>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let m_dim = dims[0];
    let n_dim = dims[1];
    let k_dim = dims[2];
    let kchunk = dims[3];
    let splits = dims[4];
    let batches = nwg.z / splits;
    let bz = wid.z / splits;
    let sz = wid.z % splits;
    let a_off = bz * dims[5];
    let b_off = bz * dims[6];
    let k_begin = sz * kchunk;
    let k_end = min(k_dim, k_begin + kchunk);
    let out_base = (sz * batches + bz) * m_dim * (n_dim / 4u);
    let kv = k_dim / 4u;
    let nv = n_dim / 4u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * BM;
    let colv0 = wid.x * BNV;

{{ decl }}

    for (var k0 = k_begin; k0 < k_end; k0 += BK) {
        // A のタイル(BM x BK)
        for (var l = tid; l < BM * (BK / 4u); l += 256u) {
            let r = l / (BK / 4u);
            let kq = l % (BK / 4u);
            let gm = row0 + r;
            let gk = k0 + kq * 4u;
            var v = vec4<f32>(0.0);
            if (gm < m_dim && gk < k_end) {
                v = a[a_off + gm * kv + gk / 4u];
            }
            let base = kq * 4u * BM + r;
            As[base] = v.x;
            As[base + BM] = v.y;
            As[base + 2u * BM] = v.z;
            As[base + 3u * BM] = v.w;
        }
        // B のタイル(BK x BN/4 個の vec4)
        for (var l = tid; l < BK * BNV; l += 256u) {
            let kr = l / BNV;
            let cv = l % BNV;
            let gk = k0 + kr;
            let gc = colv0 + cv;
            var v = vec4<f32>(0.0);
            if (gk < k_end && gc < nv) {
                v = b[b_off + gk * nv + gc];
            }
            Bs[kr * BNV + cv] = v;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < BK; kk++) {
            let ao = kk * BM + ty;
            let bo = kk * BNV + tx;
{{ body }}
        }
        workgroupBarrier();
    }

{{ store }}
}
